"""Ruído real para o "teste de sobrevivência" — base DEMAND (Zenodo 1227121).

Achado da POC: o link do Zenodo do plano NÃO é um dataset de fala, é o DEMAND, com gravações de RUÍDO
ambiente (cozinha, sala, escritório, cafeteria, trânsito, metrô, carro, parque...) em 16 canais.
Uso no projeto: misturar esse ruído aos áudios de TESTE do GSC em diferentes SNRs (relação sinal-ruído)
e medir quanto a acurácia de cada modelo cai. Nenhum modelo vê esse ruído no treino.

SNR (dB) = 10 * log10(potência da fala / potência do ruído)
    20 dB -> ruído quase imperceptível | 0 dB -> mesma potência | -5 dB -> ruído mais forte que a fala
"""
from pathlib import Path
import random

import soundfile as sf
import torch

URL_DEMAND = "https://zenodo.org/records/1227121/files/{ambiente}_16k.zip?download=1"
AMBIENTES_DEMAND = ["DKITCHEN", "DLIVING", "OOFFICE", "PCAFETER", "STRAFFIC", "TMETRO", "TCAR", "NPARK"]


def carregar_demand(pasta):
    """Lê o canal 1 de cada ambiente extraído em `pasta`. Devolve {ambiente: tensor 1D a 16 kHz}."""
    ruidos = {}
    for d in sorted(Path(pasta).iterdir()):
        canal = d / "ch01.wav"
        if d.is_dir() and canal.exists():
            wav, sr = sf.read(canal, dtype="float32")
            assert sr == 16000, f"{d.name}: esperado 16 kHz, veio {sr}"
            ruidos[d.name] = torch.from_numpy(wav)
    return ruidos


def misturar_ruido(fala, ruido, snr_db, gerador=random):
    """Soma um trecho aleatório do ruído à fala, escalado para o SNR pedido. fala: (N,), ruido: (M,), M > N."""
    n = fala.shape[-1]
    inicio = gerador.randint(0, ruido.shape[-1] - n - 1)
    trecho = ruido[inicio:inicio + n]
    p_fala = fala.pow(2).mean().clamp_min(1e-10)
    p_ruido = trecho.pow(2).mean().clamp_min(1e-10)
    #Escala o ruído para que p_fala / (escala^2 * p_ruido) = 10^(snr/10)
    escala = torch.sqrt(p_fala / (p_ruido * 10 ** (snr_db / 10)))
    return fala + escala * trecho
