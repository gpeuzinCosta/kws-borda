"""Loop de treino único para todos os modelos (loop.py), a varredura de configurações (varredura.py)
o few-shot do MSWC-pt (few_shot.py), o treino sem MFCC (sem_mfcc.py) e o script de linha de comando (treinar.py)."""
from treino.loop import treinar, treinar_uma_epoca, validar_uma_epoca, carregar_run, prever, fixar_semente, mesma_politica, mesma_arquitetura
