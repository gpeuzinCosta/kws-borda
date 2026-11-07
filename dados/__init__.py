"""Funções para gerir os datasets: download (baixar.py), MFCC (mfcc.py), GSC v2, ruído DEMAND, AudioMNIST e MSWC-pt."""
from dados.mfcc import ExtratorMFCC, calcular_n_frames, fix_length, ler_audio, to_16k_mono, SR, N_AMOSTRAS
from dados.speech_commands import (CLASSES_GSC, CLASSES_GSC12, carregar_gsc, classes_da_tarefa,
                                   criar_dataloaders, dados_falsos, ler_trecho, listar_split, locutor,
                                   locutores_em_comum, nome_cache,
                                   normalizar, rotulo, spec_augment, spec_augment_batch)
