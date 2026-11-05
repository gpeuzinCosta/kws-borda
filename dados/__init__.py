"""Funções para gerir os datasets: download (baixar.py), MFCC (mfcc.py), GSC v2, ruído DEMAND, AudioMNIST e MSWC-pt."""
from dados.mfcc import ExtratorMFCC, fix_length, ler_audio, to_16k_mono, SR, N_AMOSTRAS
from dados.speech_commands import (CLASSES_GSC, carregar_gsc, criar_dataloaders, dados_falsos,
                                      listar_split)
