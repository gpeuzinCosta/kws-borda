"""Avaliação fora do GSC: ruído real (DEMAND), AudioMNIST e MSWC-pt.

Todos usam o MESMO pipeline do treino: áudio -> 16 kHz mono, 1 s -> MFCC -> normalização com as
estatísticas do TREINO do GSC -> modelo. Nenhum desses dados é visto no treino.
"""
import random

import torch

from dados.ruido import misturar_ruido
from dados.speech_commands import normalizar


@torch.no_grad()
def logits_de_wavs(model, wavs, extrator, media, desvio, batch=256):
    """Lista de waveforms (N_AMOSTRAS,) -> logits (len(wavs), n_classes).

    media/desvio = None quando a normalização é por amostra.
    """
    device = next(model.parameters()).device
    extrator = extrator.to(device)
    model.eval()
    saidas = []
    for i in range(0, len(wavs), batch):
        x = extrator(torch.stack(wavs[i:i + batch]).to(device))
        x = normalizar(x, None if media is None else media.to(device), None if desvio is None else desvio.to(device))
        saidas.append(model(x).cpu())
    return torch.cat(saidas)


def avaliar_ruido(modelos, wavs, y, ruidos, snrs, seed=0):
    """Acurácia de cada modelo em cada SNR. Devolve {nome: {'limpo'|snr: acc}}.

    modelos: {nome: dict(model=..., extrator=..., media=..., desvio=...)}. Cada modelo usa a SUA entrada
    (o MFCC e a normalização do seu configs/<modelo>.yaml; a CNN 1D, por exemplo, usa 10 x 49).
    O MESMO trecho de ruído é usado para todos os modelos em cada SNR (comparação justa).
    """
    rng = random.Random(seed)
    res = {nome: {} for nome in modelos}
    for snr in ["limpo"] + list(snrs):
        if snr == "limpo":
            misturados = wavs
        else:
            misturados = [misturar_ruido(w, ruidos[rng.choice(list(ruidos))], snr, rng) for w in wavs]
        for nome, m in modelos.items():
            pred = logits_de_wavs(m["model"], misturados, m["extrator"], m["media"], m["desvio"]).argmax(1)
            res[nome][snr] = (pred == y).float().mean().item()
        print(snr, {n: f"{res[n][snr]*100:.1f}%" for n in res})
    return res
