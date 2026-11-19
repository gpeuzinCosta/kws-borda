"""Multilingual Spoken Words Corpus (MSWC) — português. Teste de transferência de idioma.

1,1 GB, 5.664 palavras, opus 48 kHz. Dígitos, "sim", "não", "cima", "baixo" e "vai" têm
clipes suficientes; "esquerda", "direita", "parar", "ligar", "desligar" têm poucos; "um" não existe
(o MSWC exclui palavras com menos de 3 letras).

Dois experimentos (§10.4 do relatório):
1. zero-shot mapeado: o modelo em inglês acerta o equivalente (sim->yes, cima->up)? Esperado: perto do acaso.
2. few-shot: congela a rede, troca só a última camada e treina com K exemplos por palavra (treino/few_shot.py).

O split oficial do MSWC (treino/dev/teste) NÃO separa os falantes (o mesmo falante aparece em mais de um conjunto).
Por isso dividir_por_falante faz a nossa divisão: nenhum falante aparece no treino e na avaliação do few-shot.
"""
from pathlib import Path
import random

import pandas as pd

URL_MSWC_PT = "https://mswc.mlcommons-storage.org/audio/pt.tar.gz"
#Metadados de cada clipe (SET, LINK, WORD, VALID, SPEAKER, GENDER): é daqui que vem o FALANTE
URL_SPLITS_MSWC_PT = "https://mswc.mlcommons-storage.org/splits/pt.tar.gz"

#Palavras PT equivalentes aos comandos/dígitos do GSC
PT2EN = {"sim": "yes", "não": "no", "esquerda": "left", "direita": "right", "parar": "stop",
         "zero": "zero", "um": "one", "dois": "two", "três": "three", "quatro": "four",
         "cinco": "five", "seis": "six", "sete": "seven", "oito": "eight", "nove": "nine",
         "cima": "up", "baixo": "down", "ligar": "on", "desligar": "off", "vai": "go"}


def contar_clipes(pasta_clips):
    """Quantos clipes cada palavra de PT2EN tem no corpus extraído."""
    pasta = Path(pasta_clips)
    return {w: len(list((pasta / w).iterdir())) if (pasta / w).is_dir() else 0 for w in PT2EN}


def carregar_splits(pasta_mswc):
    """Tabela de metadados dos clipes em PT (baixada pelo dados/baixar.py em <pasta_mswc>/splits)."""
    csv = next(Path(pasta_mswc, "splits").glob("**/*splits*.csv"))
    return pd.read_csv(csv)


def dividir_por_falante(splits, pasta_clips, k_max, fracao_avaliacao=0.3, max_por_falante=3, min_avaliacao=10,
                        max_treino=100, max_avaliacao=100, seed=0):
    """Divide os FALANTES (não os clipes) entre treino e avaliação do few-shot. Devolve (palavras, amostras).

    - só as palavras do PT2EN e só os clipes que existem no disco;
    - fracao_avaliacao dos falantes vão para a avaliação; os outros, para o treino (ninguém nos dois);
    - no máximo max_por_falante clipes de cada falante por palavra (um falante não domina uma palavra);
    - palavras usadas: as com pelo menos k_max clipes de treino (o maior K) e min_avaliacao de avaliação;
    - até max_treino / max_avaliacao clipes por palavra em cada conjunto (memória e tempo).
    amostras: DataFrame com WORD, SPEAKER, caminho e conjunto ('treino' | 'avaliacao').
    """
    splits = splits[splits["WORD"].isin(list(PT2EN))].copy()
    splits["caminho"] = [Path(pasta_clips) / link for link in splits["LINK"]]
    splits = splits[[c.exists() for c in splits["caminho"]]]

    #Divide os FALANTES: ninguém aparece nos dois lados
    falantes = sorted(splits["SPEAKER"].unique())
    random.Random(seed).shuffle(falantes)
    falantes_avaliacao = set(falantes[:int(fracao_avaliacao * len(falantes))])
    splits["conjunto"] = ["avaliacao" if s in falantes_avaliacao else "treino" for s in splits["SPEAKER"]]
    splits = splits.sample(frac=1, random_state=seed).groupby(["WORD", "SPEAKER"]).head(max_por_falante)

    contagem = splits.groupby(["WORD", "conjunto"]).size().unstack(fill_value=0)
    palavras = sorted(w for w, linha in contagem.iterrows()
                      if linha.get("treino", 0) >= k_max and linha.get("avaliacao", 0) >= min_avaliacao)

    amostras = []
    for w in palavras:
        for conjunto, maximo in [("treino", max_treino), ("avaliacao", max_avaliacao)]:
            linhas_w = splits[(splits["WORD"] == w) & (splits["conjunto"] == conjunto)]
            amostras.append(linhas_w.sample(min(maximo, len(linhas_w)), random_state=seed))
    return palavras, pd.concat(amostras).reset_index(drop=True)
