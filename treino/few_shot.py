"""Few-shot no MSWC-pt (§10.4 do relatório): congela a rede treinada no GSC e treina só uma head nova.

Igual aos exercícios de transfer learning da disciplina: todos os parâmetros da rede ficam com requires_grad=False,
a última camada (`head`, 35 palavras em inglês) é trocada por uma Linear nova (uma saída por palavra em português),
e só ela é treinada, com K exemplos por palavra. Se a rede aprendeu representações úteis de fala, poucos exemplos
bastam. O controle (SoMFCC) é a mesma head treinada direto sobre o MFCC, sem rede nenhuma.
"""
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dados.speech_commands import normalizar
from treino.loop import treinar_uma_epoca, validar_uma_epoca


class RedeCongelada(torch.nn.Module):
    """O modelo do GSC congelado, com uma head nova (a única parte treinável)."""

    def __init__(self, modelo_gsc, n_classes):
        super().__init__()
        self.rede = copy.deepcopy(modelo_gsc)          #cópia: o modelo original do GSC não muda

        #Congela todos os parâmetros da rede treinada no GSC
        for p in self.rede.parameters():
            p.requires_grad = False

        #Troca a head (35 palavras em inglês) por uma nova, com uma saída por palavra em português.
        #Camada nova criada do zero: por padrão requires_grad=True, então ela já nasce treinável
        self.rede.head = torch.nn.Linear(self.rede.head.in_features, n_classes)

    def train(self, mode=True):
        super().train(mode)
        self.rede.eval()                #BatchNorm e dropout da parte congelada sempre em modo de avaliação
        self.rede.head.train(mode)      #só a head nova alterna entre treino e avaliação
        return self

    def forward(self, x):
        return self.rede(x)


class SoMFCC(torch.nn.Module):
    """Controle: nenhuma rede, só uma head linear sobre o MFCC achatado (n_mfcc x n_frames valores)."""

    def __init__(self, n_entrada, n_classes):
        super().__init__()
        self.head = torch.nn.Linear(n_entrada, n_classes)

    def forward(self, x):
        return self.head(x.flatten(1))


def entrada_do_modelo(m, wavs):
    """MFCC normalizado de cada áudio, com a entrada do modelo (o mesmo que logits_de_wavs faz antes da rede).

    m: dict(model, extrator, media, desvio), o formato da §10.
    """
    with torch.no_grad():
        extrator = m["extrator"].cpu()
        x = extrator(torch.stack(wavs))
        return normalizar(x, None if m["media"] is None else m["media"].cpu(),
                          None if m["desvio"] is None else m["desvio"].cpu())


def few_shot_congelado(m, X, y, eh_treino, K, hp_few, repeticoes=5, seed=0, device=None):
    """Rede congelada + head nova treinada com K exemplos por classe. Devolve (média, desvio) da acc em %.

    X: entradas já prontas (entrada_do_modelo) | y: rótulos 0..n-1 | eh_treino: máscara dos clipes de treino.
    hp_few: dict(epocas, batch_size, learning_rate, weight_decay).
    """
    device = device or next(m["model"].parameters()).device
    n_classes = len(y.unique())
    rng = random.Random(seed)       #mesma semente para todos os modelos: os mesmos K exemplos sorteados
    idx_treino = {c: torch.where(eh_treino & (y == c))[0].tolist() for c in y.unique().tolist()}
    idx_teste = torch.where(~eh_treino)[0]
    loader_teste = DataLoader(TensorDataset(X[idx_teste], y[idx_teste]), batch_size=256)
    criterion = torch.nn.CrossEntropyLoss()
    accs = []
    for r in range(repeticoes):
        suporte = [i for idxs in idx_treino.values() for i in rng.sample(idxs, K)]
        loader_treino = DataLoader(TensorDataset(X[suporte], y[suporte]), batch_size=hp_few["batch_size"], shuffle=True)

        torch.manual_seed(seed + r)     #mesma inicialização da head para a mesma repetição
        rede = RedeCongelada(m["model"], n_classes).to(device)
        #Tudo o que não é a head: pesos congelados + buffers (média e variância acumuladas do BatchNorm)
        congelado = lambda: {k: v.detach().clone() for k, v in rede.rede.state_dict().items() if "head." not in k}
        antes = congelado()

        #O otimizador recebe só os parâmetros treináveis (a head nova)
        optimizer = torch.optim.AdamW([p for p in rede.parameters() if p.requires_grad],
                                      lr=hp_few["learning_rate"], weight_decay=hp_few["weight_decay"])
        for epoca in range(hp_few["epocas"]):
            treinar_uma_epoca(rede, loader_treino, criterion, device, optimizer)
        _, acc = validar_uma_epoca(rede, loader_teste, criterion, device)
        accs.append(acc * 100)

        #Conferência: a parte congelada não mudou (pesos e estatísticas do BatchNorm)
        depois = congelado()
        assert all(torch.equal(antes[k], depois[k]) for k in antes), "a parte congelada mudou!"
    return np.mean(accs), np.std(accs)


def tabela_few_shot(modelos, wavs, y, eh_treino, k_shots, hp_few, repeticoes=5, controle="cnn1d", device=None):
    """few_shot_congelado para cada modelo e cada K, mais o controle (head sobre o MFCC cru da entrada de `controle`).

    Devolve um DataFrame: uma linha por modelo, colunas 'K=..' (texto 'média ± desvio') e '_k..' (só a média).
    """
    import pandas as pd
    ent = modelos[controle] if controle in modelos else next(iter(modelos.values()))
    X_controle = entrada_do_modelo(ent, wavs)
    m_controle = dict(ent, model=SoMFCC(X_controle[0].numel(), len(y.unique())).to(device or "cpu"))

    linhas = []
    for nome, m in list(modelos.items()) + [("controle: MFCC cru", m_controle)]:
        X = X_controle if nome.startswith("controle") else entrada_do_modelo(m, wavs)
        linha = {"modelo": nome}
        for K in k_shots:
            media_k, desvio_k = few_shot_congelado(m, X, y, eh_treino, K, hp_few, repeticoes, device=device)
            linha[f"K={K}"] = f"{media_k:.1f} ± {desvio_k:.1f}"
            linha[f"_k{K}"] = media_k
        linhas.append(linha)
        print(nome, {k: v for k, v in linha.items() if k.startswith("K=")})
    return pd.DataFrame(linhas).set_index("modelo")
