"""Latência — hipótese (b): em modelos tão pequenos, o que custa mais, o MFCC ou a rede?

Protocolo (fixado na POC para ser justo e reprodutível):
- batch = 1 (um comando de voz por vez, como no dispositivo)
- CPU com 1 thread (simula um núcleo de um dispositivo de borda)
- aquecimento antes de medir (caches, alocações) e mediana + p95 de várias repetições
- o MFCC é medido SEPARADO da rede, e a biblioteca do MFCC é registrada (torchaudio vs librosa
  mudou a resposta na POC: com librosa o MFCC custou mais que a rede em todos os 5 modelos)
"""
import time

import numpy as np
import torch

from dados.mfcc import N_AMOSTRAS, ExtratorMFCC, mfcc_librosa


def cronometrar(fn, repeticoes=300, aquecimento=30):
    """Executa fn várias vezes. Devolve (mediana_ms, p95_ms)."""
    for _ in range(aquecimento):
        fn()
    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        fn()
        tempos.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(tempos)), float(np.percentile(tempos, 95))


@torch.no_grad()
def medir_latencia(model, extrator=None, threads=1, repeticoes=300, aquecimento=30, com_librosa=False):
    """Mede MFCC e rede separadamente na CPU. Devolve dicionário com mediana/p95 em ms."""
    threads_antes = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        extrator = (extrator or ExtratorMFCC()).cpu().eval()
        model = model.cpu().eval()
        wav = torch.randn(1, N_AMOSTRAS) * 0.1
        feat = extrator(wav)

        mfcc_med, mfcc_p95 = cronometrar(lambda: extrator(wav), repeticoes, aquecimento)
        rede_med, rede_p95 = cronometrar(lambda: model(feat), repeticoes, aquecimento)
        r = dict(mfcc_ms=mfcc_med, mfcc_p95_ms=mfcc_p95, rede_ms=rede_med, rede_p95_ms=rede_p95,
                 mfcc_sobre_rede=mfcc_med / rede_med, threads=threads, mfcc_lib="torchaudio")
        if com_librosa:
            wav_np = wav.numpy()[0]
            r["mfcc_librosa_ms"], r["mfcc_librosa_p95_ms"] = cronometrar(
                lambda: mfcc_librosa(wav_np), repeticoes, aquecimento)
        return r
    finally:
        torch.set_num_threads(threads_antes)
