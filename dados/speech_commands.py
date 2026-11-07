"""Google Speech Commands v2 (GSC v2) — o dataset núcleo do projeto (treino, validação e teste).

- 105.835 áudios de 1 s, 16 kHz, 35 palavras, ~2.600 locutores.
- Split OFICIAL por locutor (validation_list.txt / testing_list.txt): o mesmo locutor nunca aparece
  em treino e teste, então medimos generalização para vozes novas.
- Lemos os .wav com soundfile direto da pasta extraída em vez de
  torchaudio.datasets.SPEECHCOMMANDS — mesmo conteúdo, sem depender do backend de áudio do torchaudio.
- O MFCC de todo o dataset é calculado UMA vez e salvo em cache (.pt): treinar fica muito mais rápido
  e todos os modelos recebem exatamente a mesma entrada.

Duas TAREFAS (hp["tarefa"]):
- "gsc35": as 35 palavras. Necessária para o AudioMNIST (dígitos) e o MSWC-pt (mapeamento PT -> EN).
- "gsc12": o recorte padrão da literatura (Hello Edge, KWT) — 10 comandos + _unknown_ + _silence_.
  Torna os números comparáveis aos artigos. Montagem trazida do notebook do Luciano
  (notebooks/experimentos/Projeto_MLP.ipynb):
    * _unknown_: sorteio das outras 25 palavras, na mesma quantidade média de cada comando;
    * _silence_: trechos de 1 s dos ruídos de _background_noise_, com volume reduzido a 10%.
"""
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.auto import tqdm

from dados.mfcc import N_AMOSTRAS, ExtratorMFCC, calcular_n_frames, ler_audio

URL_GSC = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

#As 35 palavras do GSC v2, em ordem alfabética (índice = rótulo da classe na tarefa gsc35)
CLASSES_GSC = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero",
]

#Tarefa de 12 classes (padrão Hello Edge / KWT)
COMANDOS_12 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
CLASSES_GSC12 = COMANDOS_12 + ["_unknown_", "_silence_"]

TAREFAS = {"gsc35": CLASSES_GSC, "gsc12": CLASSES_GSC12}

VOLUME_SILENCIO = 0.1     #mesmo fator do notebook do Luciano
SEMENTE_SPLIT = 42        #o sorteio de unknown/silence é fixo: todos usam os mesmos arquivos


def classes_da_tarefa(tarefa):
    if tarefa not in TAREFAS:
        raise KeyError(f"Tarefa '{tarefa}' desconhecida. Opções: {list(TAREFAS)}")
    return TAREFAS[tarefa]


def rotulo(rel, tarefa):
    """Caminho relativo -> índice da classe na tarefa. Ex.: 'cat/x.wav' -> _unknown_ na gsc12."""
    palavra = rel.split("/")[0]
    if tarefa == "gsc12" and palavra not in COMANDOS_12 and palavra != "_silence_":
        palavra = "_unknown_"
    return classes_da_tarefa(tarefa).index(palavra)


def _listar_35(raiz):
    raiz = Path(raiz)
    ler = lambda f: set((raiz / f).read_text().split())
    val, teste = ler("validation_list.txt"), ler("testing_list.txt")
    split = {"treino": [], "val": [], "teste": []}
    for c in CLASSES_GSC:
        for f in sorted(p.name for p in (raiz / c).iterdir()):
            rel = f"{c}/{f}"
            split["val" if rel in val else "teste" if rel in teste else "treino"].append(rel)
    return split


def _montar_12(raiz, split35):
    """Filtra os 10 comandos e adiciona unknown (sorteado) e silence (trechos de ruído).

    Os trechos de silêncio são representados como '_silence_/<arquivo de ruído>@<início>', e o
    `ler_trecho` sabe recortá-los. Assim tudo continua sendo uma lista de caminhos (e o cache funciona).
    """
    rng = random.Random(SEMENTE_SPLIT)
    pasta_ruido = Path(raiz) / "_background_noise_"
    ruidos = {f.name: sf.info(f).frames for f in sorted(pasta_ruido.glob("*.wav"))}

    split = {}
    for nome, arquivos in split35.items():
        comandos = [r for r in arquivos if r.split("/")[0] in COMANDOS_12]
        outros = [r for r in arquivos if r.split("/")[0] not in COMANDOS_12]
        por_classe = len(comandos) // len(COMANDOS_12)

        unknown = rng.sample(outros, min(por_classe, len(outros)))
        silencio = []
        for _ in range(por_classe):
            arq = rng.choice(list(ruidos))
            inicio = rng.randint(0, ruidos[arq] - N_AMOSTRAS)
            silencio.append(f"_silence_/{arq}@{inicio}")
        split[nome] = comandos + unknown + silencio
    return split


def listar_split(raiz, tarefa="gsc35"):
    """Separa os arquivos em treino/val/teste (listas oficiais) para a tarefa. Devolve caminhos relativos."""
    split35 = _listar_35(raiz)
    return split35 if tarefa == "gsc35" else _montar_12(raiz, split35)


#O ID do locutor é o que vem antes de '_nohash_' no nome do arquivo
def locutor(rel):
    return rel.split('/')[1].split('_nohash_')[0]


def locutores_em_comum(split):
    """Conferência do split oficial: quantos locutores aparecem em mais de um conjunto (esperado: 0 e 0).
    Os trechos de silêncio da gsc12 não têm locutor e ficam de fora."""
    locutores = {k: {locutor(r) for r in v if not r.startswith('_silence_/')} for k, v in split.items()}
    return {'treino_e_teste': len(locutores['treino'] & locutores['teste']),
            'treino_e_val': len(locutores['treino'] & locutores['val'])}


def ler_trecho(raiz, rel):
    """Lê um áudio do GSC ou um trecho de silêncio ('_silence_/arquivo.wav@inicio')."""
    if rel.startswith("_silence_/"):
        arq, inicio = rel.split("/", 1)[1].split("@")
        wav, _ = sf.read(Path(raiz) / "_background_noise_" / arq, dtype="float32",
                         start=int(inicio), frames=N_AMOSTRAS)
        return VOLUME_SILENCIO * torch.from_numpy(wav)
    return ler_audio(Path(raiz) / rel)


def extrair_mfcc(raiz, arquivos, extrator, tarefa, caminho_cache=None, batch=256):
    """Lê os áudios e calcula o MFCC de todos. Se houver cache, só carrega. Devolve (X, y)."""
    if caminho_cache is not None and Path(caminho_cache).exists():
        return torch.load(caminho_cache)
    X, y = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(arquivos), batch), desc="MFCC"):
            lote = arquivos[i:i + batch]
            X.append(extrator(torch.stack([ler_trecho(raiz, rel) for rel in lote])))
            y += [rotulo(rel, tarefa) for rel in lote]
    X, y = torch.cat(X), torch.tensor(y)
    if caminho_cache is not None:
        torch.save((X, y), caminho_cache)
    return X, y


def normalizar(x, media=None, desvio=None):
    """Normaliza o MFCC (..., n_mfcc, n_frames).

    - global (media/desvio dados): estatísticas por coeficiente calculadas no TREINO. O dispositivo
      precisa guardar esses 2 x n_mfcc números.
    - por amostra (media=None): cada áudio é normalizado pela própria média/desvio ao longo do tempo
      (abordagem do notebook do Luciano). Não precisa guardar nada e se adapta ao volume do microfone.
    """

    if media is None:
        #por_amostra: cada áudio usa a PRÓPRIA média e desvio ao longo do tempo
        #O 1e-6 evita dividir por zero num áudio perfeitamente constante
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    #global: (x - mean) / std
    return (x - media) / desvio


class DatasetMFCC(Dataset):

    def __init__(self, X, y, media, desvio, aumentar=False):
        super().__init__()

        #Normaliza tudo de uma vez aqui, e não um áudio por __getitem__
        self.X = normalizar(X, media, desvio)
        self.y = y
        self.aumentar = aumentar    #SpecAugment ligado ou desligado

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """Retorna (mfcc: torch.Tensor (40, 101), label: int).

        idx pode ser um índice ou uma LISTA de índices: com a lista (o BatchSampler do criar_dataloaders),
        o self.X[idx] devolve o batch inteiro (B, n_mfcc, n_frames) com um único fatiamento.
        """
        x = self.X[idx]

        #Aumento de dados sorteado a CADA vez que o áudio é pedido: na época seguinte as faixas mudam
        #(nos DataLoaders do projeto fica desligado: o aumento é aplicado no batch, na GPU, pelo spec_augment_batch)
        if self.aumentar:
            x = spec_augment(x)

        return x, self.y[idx]


def spec_augment(x, max_tempo=10, max_coef=7):
    """x: (n_mfcc, n_frames) de UM áudio. Zera uma faixa de janelas e uma faixa de coeficientes."""

    #Trabalha numa CÓPIA. Sem isso, o x[...] = 0 apagaria o trecho direto no X guardado na memória,
    #e o áudio ficaria mutilado para sempre, em todas as épocas seguintes
    x = x.clone()
    n_coef, n_frames = x.shape

    #Com poucos coeficientes, zerar 7 apagaria quase tudo
    max_coef = min(max_coef, n_coef // 4)

    #Faixa no tempo
    #Largura sorteada de 0 a max_tempo janelas (o () pede um sorteio escalar; o .item() vira int do Python)
    t = torch.randint(0, max_tempo + 1, ()).item()
    #de 0 até o último ponto em que a faixa ainda cabe dentro das 101 janelas
    t0 = torch.randint(0, n_frames - t + 1, ()).item()
    x[:, t0:t0 + t] = 0     #zera todos os coeficientes nessas janelas

    #Faixa na frequência
    f = torch.randint(0, max_coef + 1, ()).item()
    f0 = torch.randint(0, n_coef - f + 1, ()).item()
    x[f0:f0 + f, :] = 0     #zera essas linhas em todas as janelas

    return x


def spec_augment_batch(x, max_tempo=10, max_coef=7):
    """x: (B, n_mfcc, n_frames), já na GPU. Mesmo sorteio da spec_augment, mas cada áudio do batch
    recebe as PRÓPRIAS faixas, todas calculadas de uma vez com operações de tensor (sem laço em Python)."""
    B, n_coef, n_frames = x.shape
    max_coef = min(max_coef, n_coef // 4)

    #Faixa no tempo: uma largura e um início sorteados POR ÁUDIO (vetores de tamanho B)
    t = torch.randint(0, max_tempo + 1, (B,), device=x.device)
    #Início uniforme entre 0 e n_frames - t (a faixa sempre cabe): rand em [0, 1) x (n_frames - t + 1), arredondado para baixo
    t0 = (torch.rand(B, device=x.device) * (n_frames - t + 1)).floor().long()

    #Faixa na frequência (mesma lógica)
    f = torch.randint(0, max_coef + 1, (B,), device=x.device)
    f0 = (torch.rand(B, device=x.device) * (n_coef - f + 1)).floor().long()

    #Máscaras: compara a posição de cada janela/coeficiente com o intervalo sorteado de cada áudio
    janelas = torch.arange(n_frames, device=x.device)                             # (n_frames,)
    coefs = torch.arange(n_coef, device=x.device)                                 # (n_coef,)
    mascara_tempo = (janelas >= t0[:, None]) & (janelas < (t0 + t)[:, None])      # (B, n_frames)
    mascara_coef = (coefs >= f0[:, None]) & (coefs < (f0 + f)[:, None])           # (B, n_coef)

    #masked_fill devolve um tensor NOVO: o batch original não é alterado (o papel do .clone() da versão antiga)
    x = x.masked_fill(mascara_tempo[:, None, :], 0)    # a mesma faixa de tempo em todos os coeficientes
    x = x.masked_fill(mascara_coef[:, :, None], 0)     # a mesma faixa de coeficientes em todas as janelas
    return x


def dados_falsos(hp, n=256):
    """MFCCs aleatórios com o formato certo — para testar o pipeline sem baixar 2,3 GB."""
    X = torch.randn(n, hp["n_mfcc"], calcular_n_frames(hp))
    y = torch.randint(0, hp["n_classes"], (n,))
    return {"treino": (X, y), "val": (X[: n // 4], y[: n // 4]), "teste": (X[: n // 4], y[: n // 4])}


def nome_cache(hp):
    """Identifica a combinação tarefa + MFCC (caches diferentes não se misturam)."""
    centro = "c" if hp.get("center", True) else "nc"
    return f"{hp['tarefa']}_{hp['n_mfcc']}x{calcular_n_frames(hp)}_{hp['janela_ms']}-{hp['passo_ms']}ms{centro}"


def carregar_gsc(hp, raiz, pasta_cache):
    """MFCC de treino/val/teste do GSC (com cache). Devolve {'treino': (X, y), 'val': ..., 'teste': ...}."""
    extrator = ExtratorMFCC.de_hparams(hp)
    split = listar_split(raiz, hp["tarefa"])
    print(hp["tarefa"], {k: len(v) for k, v in split.items()})
    return {k: extrair_mfcc(raiz, v, extrator, hp["tarefa"], Path(pasta_cache) / f"gsc_{k}_{nome_cache(hp)}.pt")
            for k, v in split.items()}


def criar_dataloaders(hp, dados):
    """Normaliza e cria os DataLoaders. Devolve (loaders, (media, desvio)).

    Mesma lógica do DataModuleGSC do playground da CNN 1D:
    - hp['fracao'] < 1 usa só uma fração do TREINO (validação e teste são sempre completos);
    - com normalizacao='global', media/desvio vêm SÓ do treino (medir na validação seria vazamento);
      com normalizacao='por_amostra', devolve (None, None);
    - o SpecAugment NÃO entra aqui: é aplicado no batch, na GPU, dentro do treinar_uma_epoca (spec_augment_batch).
    """
    X_treino, y_treino = dados["treino"]

    #Fração do treino. Val e teste são sempre completos
    if hp.get("fracao", 1.0) < 1:
        #Semente fixa: a mesma fração dá sempre os mesmos exemplos
        gerador = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(y_treino), generator=gerador)[:int(len(y_treino) * hp["fracao"])]
        X_treino, y_treino = X_treino[idx], y_treino[idx]

    #Estatísticas medidas SÓ no treino
    #A validação e o teste usam as estatísticas do TREINO
    if hp.get("normalizacao", "global") == "global":
        media = X_treino.mean(dim=(0, 2), keepdim=True)[0]      #(n_mfcc, 1): uma média por coeficiente
        desvio = X_treino.std(dim=(0, 2), keepdim=True)[0]
    else:
        #por_amostra: nada a guardar
        media = None
        desvio = None

    conjuntos = {"treino": DatasetMFCC(X_treino, y_treino, media, desvio)}
    for nome in dados:
        if nome != "treino":
            conjuntos[nome] = DatasetMFCC(*dados[nome], media, desvio)

    #O sampler entrega a LISTA de índices do batch, e o DatasetMFCC devolve o batch inteiro com um fatiamento
    #(self.X[lista]) em vez de 128 chamadas do __getitem__ + empilhamento. batch_size=None desliga o
    #empilhamento automático do DataLoader, porque o batch já vem pronto
    loaders = {}
    for nome, ds in conjuntos.items():
        if nome == "treino":
            amostrador = BatchSampler(RandomSampler(ds), batch_size=hp["batch_size"], drop_last=False)
        else:
            amostrador = BatchSampler(SequentialSampler(ds), batch_size=hp["batch_size"], drop_last=False)
        loaders[nome] = DataLoader(ds, batch_size=None, sampler=amostrador)
    return loaders, (media, desvio)
