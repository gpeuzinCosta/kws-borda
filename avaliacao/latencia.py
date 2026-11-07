"""Latência — hipótese (b): em modelos tão pequenos, o que custa mais, o MFCC ou a rede?

Protocolo (para ser justo e reprodutível), trazido do playground da CNN 1D:
- batch = 1 (um comando de voz por vez, como no dispositivo)
- CPU com 1 thread (simula um núcleo de um dispositivo de borda)
- aquecimento antes de medir (caches, alocações) e mediana + p95 de várias repetições
- MFCC, normalização e rede medidos SEPARADOS, em várias rodadas; a oscilação da razão
  (pré-processamento / rede) entre as rodadas diz se a medida é confiável
- opcional: o MFCC também com librosa, porque a implementação pode mudar a resposta da hipótese (b)
"""
import copy
import platform
import time

import numpy as np
import torch

from dados.mfcc import N_AMOSTRAS, ExtratorMFCC, mfcc_librosa
from dados.speech_commands import normalizar


def cronometrar(fn, repeticoes=300, aquecimento=30):
    #Aquecimento: as primeiras chamadas são mais lentas (caches vazios, alocação de memória).
    #Não entram na medida
    for _ in range(aquecimento):
        fn()

    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()     #relógio de alta resolução, feito para medir intervalos curtos
        fn()
        tempos.append((time.perf_counter() - t0) * 1e3)    #segundos -> milissegundos

    #Mediana
    #O p95 diz "em 95% das vezes foi mais rápido que isso"
    return float(np.median(tempos)), float(np.percentile(tempos, 95))


LIMITE_INSTAVEL = 0.30   #escolha minha, ajustável: (máx - mín) / mediana da razão entre as rodadas


def nome_cpu():
    """Modelo da CPU desta medida (o Colab muda de máquina entre sessões)."""
    try:
        with open('/proc/cpuinfo') as f:
            for linha in f:
                if linha.startswith('model name'):
                    return linha.split(':', 1)[1].strip()
    except OSError:
        pass   #fora do Linux (ex.: Mac) não existe /proc/cpuinfo
    return platform.processor() or 'desconhecida'


@torch.no_grad()
def medir_latencia(model, hp, media, desvio, rodadas=5, repeticoes=300, aquecimento=30, tentativas=3,
                   com_librosa=False):
    """Mede MFCC, normalização e rede, separadamente, na CPU. Devolve um dicionário em ms.

    media/desvio: as estatísticas da normalização global ((n_mfcc, 1)) ou None (por_amostra).
    """

    #CPU com 1 thread (simula um núcleo de dispositivo de borda) e batch 1 (um comando por vez)
    threads_antes = torch.get_num_threads()
    torch.set_num_threads(hp.get('latencia_threads', 1))

    try:    #sem o try/finally, um erro no meio deixaria o notebook inteiro com 1 thread
        #Cópia na CPU
        modelo_cpu = copy.deepcopy(model).cpu().eval()
        ext_cpu = ExtratorMFCC.de_hparams(hp).cpu().eval()
        if media is not None:
            media, desvio = media.cpu(), desvio.cpu()

        wav = torch.randn(1, N_AMOSTRAS) * 0.1                 # (1, 16000): um áudio de 1 s
        feat = ext_cpu(wav)                                    # (1, n_mfcc, n_frames)
        feat_norm = normalizar(feat, media, desvio)            #a entrada que a rede realmente recebe
        feat_x8 = feat_norm.repeat(8, 1, 1)                    #8 áudios iguais, para o teste de overhead

        melhor = None
        for tentativa in range(1, tentativas + 1):
            med = {'mfcc': [], 'norm': [], 'rede': [], 'rede_x8': []}
            p95 = {'mfcc': [], 'norm': [], 'rede': []}

            #As 3 etapas são medidas DENTRO de cada rodada (e não 5 rodadas de uma etapa, depois 5 de outra):
            #se a máquina ficar mais lenta no meio, o efeito cai nas 3 igualmente e a razão se preserva
            for _ in range(rodadas):
                m, p = cronometrar(lambda: ext_cpu(wav), repeticoes, aquecimento)
                med['mfcc'].append(m); p95['mfcc'].append(p)

                m, p = cronometrar(lambda: normalizar(feat, media, desvio), repeticoes, aquecimento)
                med['norm'].append(m); p95['norm'].append(p)

                m, p = cronometrar(lambda: modelo_cpu(feat_norm), repeticoes, aquecimento)
                med['rede'].append(m); p95['rede'].append(p)

                m, _ = cronometrar(lambda: modelo_cpu(feat_x8), repeticoes, aquecimento)
                med['rede_x8'].append(m)

            #A razão da hipótese (b), rodada a rodada. A variação dela entre as rodadas é o que diz se a medida é confiável
            razoes = [(a + b) / c for a, b, c in zip(med['mfcc'], med['norm'], med['rede'])]
            oscilacao = (max(razoes) - min(razoes)) / np.median(razoes)

            #Guarda a tentativa mais estável vista até agora
            if melhor is None or oscilacao < melhor['oscilacao']:
                melhor = dict(med=med, p95=p95, razoes=razoes, oscilacao=oscilacao)

            if oscilacao <= LIMITE_INSTAVEL:
                break    #estável: não precisa repetir

        #MFCC com librosa (mesma configuração), para comparar o custo de IMPLEMENTAÇÃO do MFCC
        librosa_ms = None
        if com_librosa:
            wav_np = wav.numpy()[0]
            cfg = ext_cpu.config
            librosa_ms, librosa_p95 = cronometrar(
                lambda: mfcc_librosa(wav_np, n_mfcc=cfg['n_mfcc'], n_fft=cfg['n_fft'], hop=cfg['hop'],
                                     n_mels=cfg['n_mels'], f_min=cfg['f_min'], f_max=cfg['f_max']),
                repeticoes, aquecimento)
    finally:
        torch.set_num_threads(threads_antes)

    med, p95, razoes = melhor['med'], melhor['p95'], melhor['razoes']
    estavel = melhor['oscilacao'] <= LIMITE_INSTAVEL
    if not estavel:
        print(f'AVISO: latência instável (razão oscilou {melhor["oscilacao"]*100:.0f}% entre as rodadas, '
              f'limite {LIMITE_INSTAVEL*100:.0f}%). Rode de novo com a máquina ociosa antes de usar este número')

    mfcc_ms = float(np.median(med['mfcc']))
    norm_ms = float(np.median(med['norm']))
    rede_ms = float(np.median(med['rede']))
    rede_x8_ms = float(np.median(med['rede_x8']))

    r = dict(mfcc_ms=mfcc_ms, mfcc_p95_ms=float(np.median(p95['mfcc'])),
             norm_ms=norm_ms, norm_p95_ms=float(np.median(p95['norm'])),
             rede_ms=rede_ms, rede_p95_ms=float(np.median(p95['rede'])),
             preproc_ms=mfcc_ms + norm_ms,                              #pré-processamento = MFCC + normalização
             preproc_sobre_rede=float(np.median(razoes)),               #a razão da hipótese (b)
             razao_min=min(razoes), razao_max=max(razoes),
             oscilacao=float(melhor['oscilacao']), estavel=estavel, tentativas_feitas=tentativa,
             rede_x8_ms=rede_x8_ms, fator_x8=rede_x8_ms / rede_ms,      #teste de overhead
             rodadas=rodadas, cpu=nome_cpu())
    if librosa_ms is not None:
        r['mfcc_librosa_ms'], r['mfcc_librosa_p95_ms'] = librosa_ms, librosa_p95
    return r
