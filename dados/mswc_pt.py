"""Multilingual Spoken Words Corpus (MSWC) — português. Teste de transferência de idioma.

Achado da POC: 1,1 GB, 5.664 palavras, opus 48 kHz. Dígitos, "sim", "não", "cima", "baixo" e "vai" têm
clipes suficientes; "esquerda", "direita", "parar", "ligar", "desligar" têm poucos; "um" não existe
(o MSWC exclui palavras com menos de 3 letras).

Dois experimentos:
1. zero-shot mapeado: o modelo em inglês acerta o equivalente (sim->yes, cima->up)? Esperado: perto do acaso.
2. few-shot / cabeça linear: congela a rede, troca só a última camada e treina com K exemplos por palavra.
   Na POC isso chegou a ~85% com 20 exemplos/palavra.
"""
from pathlib import Path
import random

URL_MSWC_PT = "https://mswc.mlcommons-storage.org/audio/pt.tar.gz"

#Palavras PT equivalentes aos comandos/dígitos do GSC
PT2EN = {"sim": "yes", "não": "no", "esquerda": "left", "direita": "right", "parar": "stop",
         "zero": "zero", "um": "one", "dois": "two", "três": "three", "quatro": "four",
         "cinco": "five", "seis": "six", "sete": "seven", "oito": "eight", "nove": "nine",
         "cima": "up", "baixo": "down", "ligar": "on", "desligar": "off", "vai": "go"}


def contar_clipes(pasta_clips):
    """Quantos clipes cada palavra de PT2EN tem no corpus extraído."""
    pasta = Path(pasta_clips)
    return {w: len(list((pasta / w).iterdir())) if (pasta / w).is_dir() else 0 for w in PT2EN}


def listar_mswc_pt(pasta_clips, min_clipes=50, por_palavra=200, seed=0):
    """Seleciona as palavras com clipes suficientes. Devolve (palavras, arquivos, rótulos)."""
    rng = random.Random(seed)
    contagem = contar_clipes(pasta_clips)
    palavras = [w for w, c in contagem.items() if c >= min_clipes]
    arquivos, rotulos = [], []
    for i, w in enumerate(palavras):
        fs = sorted((Path(pasta_clips) / w).iterdir())
        fs = rng.sample(fs, min(por_palavra, len(fs)))
        arquivos += fs
        rotulos += [i] * len(fs)
    return palavras, arquivos, rotulos


def avaliar_cabeca_linear(*args, **kwargs):
    """TODO (seção 11 do notebook): backbone congelado + LogisticRegression com K exemplos por palavra.

    Referência pronta: função `mswc` em kws-borda-poc/eval_external.py. Pendência: usar split por falante.
    """
    raise NotImplementedError("Avaliação few-shot do MSWC-pt ainda não implementada (ver docstring).")
