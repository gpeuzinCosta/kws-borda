"""Entrada SEM MFCC: o áudio cru (16000 amostras) entra direto na rede.

O DataLoader entrega (B, 1, 16000) (dados/mfcc.py, ExtratorOnda). Uma FRENTE transforma isso numa "matriz"
(B, canais, janelas), o mesmo formato do MFCC, e a rede de qualquer família (MLP, CNN 1D, ResNet, DS-CNN, Atenção)
roda sem mudança. Duas frentes:

- 'patches' (sem nada aprendido): corta o áudio em pedaços de `onda_passo_ms` (10 ms = 160 amostras), como os
  patches do ViT. (B, 1, 16000) -> (B, 160, 100). Na Atenção, cada pedaço vira um TOKEN: Linear(160 -> d_model).
  O áudio inteiro entra, e a rede aprende sozinha o que o MFCC faria;
- 'conv' (um "MFCC aprendido"): Conv1d(1, onda_filtros, janela, stride=passo) + BatchNorm + ReLU.
  Com 40 filtros de 25 ms e passo de 10 ms: (B, 1, 16000) -> (B, 40, 98), parecido com o MFCC 40x101.

Antes da frente, cada áudio é normalizado pela própria média e desvio (a mesma ideia da normalização por amostra
do MFCC): o volume do microfone não muda a entrada.

Não dá para usar um token por AMOSTRA na Atenção: com 16000 tokens, a autoatenção compara todos os pares
(16000² = 256 milhões por camada, por áudio), o que não cabe na memória da GPU.
"""
import torch


def formato_frente(hp):
    """(canais, janelas) que a frente entrega para a rede da família."""
    n = int(hp["sample_rate"] * hp["duracao_s"])
    janela = int(hp["sample_rate"] * hp.get("onda_janela_ms", 25) / 1000)
    passo = int(hp["sample_rate"] * hp.get("onda_passo_ms", 10) / 1000)
    if hp.get("onda_frente", "conv") == "patches":
        return passo, n // passo
    if hp.get("onda_center", False):
        return hp.get("onda_filtros", 40), 1 + n // passo          #bordas preenchidas, como o MFCC com center=True
    return hp.get("onda_filtros", 40), 1 + (n - janela) // passo


class FrenteOnda(torch.nn.Module):
    """(B, 1, N) de áudio cru -> (B, canais, janelas). Ver o docstring do módulo."""

    def __init__(self, modo="conv", filtros=40, janela=400, passo=160, center=False):
        super().__init__()
        self.modo, self.passo = modo, passo
        if modo == "conv":
            #center=True: preenche janela/2 amostras em cada borda, como o MFCC do torchaudio (mesmo nº de janelas)
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(1, filtros, kernel_size=janela, stride=passo, padding=janela // 2 if center else 0,
                                bias=False),
                torch.nn.BatchNorm1d(filtros),
                torch.nn.ReLU(),
            )
        elif modo != "patches":
            raise ValueError(f"onda_frente '{modo}' desconhecida: use 'conv' ou 'patches'")

    def forward(self, x):
        x = x.float()                     #o cache do áudio cru é float16 (metade da memória)

        #Normalização por amostra do sinal: média 0 e desvio 1 em cada áudio
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)

        if self.modo == "patches":
            B, _, N = x.shape
            n_janelas = N // self.passo
            #(B, 1, N) -> (B, janelas, passo) -> (B, passo, janelas): cada coluna é um pedaço de 10 ms
            return x[..., :n_janelas * self.passo].reshape(B, n_janelas, self.passo).transpose(1, 2)
        return self.conv(x)


class ModeloOnda(torch.nn.Module):
    """FrenteOnda + a rede de uma família. A `head` é a da rede (few-shot do MSWC troca a head)."""

    def __init__(self, frente, rede):
        super().__init__()
        self.frente = frente
        self.rede = rede

    @property
    def head(self):
        return self.rede.head

    def __setattr__(self, nome, valor):
        #Trocar a head (few-shot) troca a head da REDE. Sem isto, o nn.Module registraria um submódulo novo
        #chamado head, que o forward nunca usa
        if nome == "head" and "rede" in self._modules:
            setattr(self.rede, "head", valor)
            return
        super().__setattr__(nome, valor)

    def forward(self, x):
        return self.rede(self.frente(x))


def embrulhar_onda(cls, hp, extras):
    """Cria a rede da família `cls` para a saída da frente e devolve ModeloOnda(frente, rede)."""
    canais, janelas = formato_frente(hp)
    sr = hp["sample_rate"]
    frente = FrenteOnda(hp.get("onda_frente", "conv"), hp.get("onda_filtros", 40),
                        int(sr * hp.get("onda_janela_ms", 25) / 1000), int(sr * hp.get("onda_passo_ms", 10) / 1000),
                        center=hp.get("onda_center", False))
    rede = cls(n_classes=hp["n_classes"], n_mfcc=canais, n_frames=janelas, **extras)
    return ModeloOnda(frente, rede)
