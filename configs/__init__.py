"""configs — hiperparâmetros, pastas do projeto e device.

Os hiperparâmetros são dois dicionários Python, no mesmo formato da seção 2 do notebook principal:
- HYPERPARAMETERS: o que é comum a TODOS os modelos (dados, MFCC, treino, avaliação). É o que deixa a comparação justa;
- HP_MODELOS: só o que muda em cada modelo (a arquitetura e, se for o caso, o próprio MFCC).

    hp = carregar_config("dscnn")
    hp["batch_size"], hp["learning_rate"], hp["canais"] ...
"""
import copy
import os
from pathlib import Path

#configs/__init__.py -> a pasta configs/ e a raiz do repositório
PASTA_CONFIGS = Path(__file__).resolve().parent
RAIZ_REPO = PASTA_CONFIGS.parent


#Comum a TODOS os modelos: mesmos dados, mesmo MFCC, mesmo loop de treino e mesmas métricas
HYPERPARAMETERS = {
    #Dados
    'dataset': 'speech_commands_v2',
    'tarefa': 'gsc35',          #'gsc35': as 35 palavras (necessária para AudioMNIST e MSWC-pt) | 'gsc12': 10 comandos + unknown + silence
    'sample_rate': 16000,
    'duracao_s': 1.0,

    #MFCC "padrao": 40 coef, janela 30 ms, passo 10 ms, center -> entrada (B, 40, 101)
    #Preset "hello_edge" (PRESETS_MFCC): 10 coef, 40 ms, 20 ms, sem center -> (B, 10, 49)
    'n_mfcc': 40,
    'janela_ms': 30,
    'passo_ms': 10,
    'center': True,
    'n_mels': 40,
    'f_min': 20,
    'f_max': 7600,
    #'por_amostra': cada áudio pela sua própria média/desvio (remove o volume do microfone; ver §10.2 do relatório e
    #§4.2-4.5 do playground de generalização) | 'global': média/desvio por coeficiente medidos no TREINO
    'normalizacao': 'por_amostra',
    'spec_augment': True,       #máscara de tempo e de coeficientes, igual para todos
    'fracao': 1.0,              #fração do TREINO usada (1.0 = tudo). Validação e teste são sempre completos

    #Treino
    'epocas': 50,               #TETO de épocas: o early stop (paciencia) decide quando parar
    'batch_size': 128,
    'learning_rate': 0.003,
    'weight_decay': 0.01,
    'otimizador': 'adamw',
    'scheduler': 'plateau',     #'plateau' (ReduceLROnPlateau, combina com o early stop) | 'onecycle' (usado nos playgrounds)
    'label_smoothing': 0.1,
    #Early stop: a mesma política da §2 do relatório. Os playgrounds da CNN 1D e da ResNet usaram 20 épocas com
    #OneCycle e sem early stop (para reproduzir: epocas=20, scheduler='onecycle', paciencia=None)
    'paciencia': 5,             #para após N épocas sem melhorar a acc de validação (None = desliga)
    'min_delta': 0.0005,        #melhora mínima (0,05 p.p.) para zerar a contagem da paciência
    'plateau_fator': 0.5,       #ReduceLROnPlateau: lr x fator quando a validação estagna
    'plateau_paciencia': 3,     #épocas sem melhora antes de reduzir o lr
    'seeds': [0, 1, 2],         #3 rodadas independentes para média ± desvio
    'num_workers': 0,           #MFCC já está em memória: processos extras só adicionam overhead

    #Avaliação
    'latencia_repeticoes': 300,
    'latencia_aquecimento': 30,
    'latencia_threads': 1,      #simula um dispositivo restrito (1 núcleo)
    'snrs_db': [20, 10, 5, 0, -5],
    'margem_pp': 2.0,           #critério do projeto: um modelo menor "serve" se ficar até 2 p.p. abaixo do baseline (validação)
    'acc_minima': 90.0,         #acurácia mínima (%) no teste para um reconhecimento "funcional" (hipótese a)

    #Generalização (§10 do relatório)
    'n_ruido': None,                    #áudios de teste do GSC no DEMAND e no volume (None = o teste inteiro)
    'snrs_ambiente': [5, 0, -5],        #níveis de ruído da análise por ambiente
    'snr_matriz': -5,                   #nível das matrizes de confusão com ruído
    'ganhos_db': [-40, -30, -20, -10, 0, 10],   #ganhos aplicados ao áudio no teste de volume
    'por_falante_audiomnist': 100,      #gravações por falante do AudioMNIST (60 falantes; None = todas)
    'k_shots': [1, 3, 5, 10, 20],       #MSWC-pt: exemplos de treino por palavra no few-shot
    'repeticoes_few': 5,                #sorteios dos K exemplos (média ± desvio)
    'hp_few': {'epocas': 50, 'batch_size': 16, 'learning_rate': 1e-2, 'weight_decay': 1e-2},   #treino da head nova
}


#Só o que MUDA em cada modelo
HP_MODELOS = {
    #Todas as famílias usam o VENCEDOR da sua varredura (§8): o MENOR modelo (parâmetros, depois MACs) até 2 p.p. abaixo
    #do baseline 40x101 na validação. Números da §8 (normalização por amostra, early stop 50/5, 3 seeds)

    #MLP (§8.1): hello_edge + hidden 64. 42.019 params (41,0 KB int8), 41.792 MACs,
    #acc val 72,54 ± 2,04% (baseline 74,17 ± 0,58%, −1,64 p.p.). Nenhum MLP passa da acurácia mínima de 90%
    'mlp': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,     #MFCC hello_edge: 10 x 49
            'hidden': 64, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False},

    #CNN 1D (§8.2): hello_edge + canais 48. 24.275 params (23,7 KB int8), 383.280 MACs,
    #acc val 92,09 ± 0,24% (baseline 93,82 ± 0,10%, −1,73 p.p.)
    'cnn1d': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,   #MFCC hello_edge: 10 x 49
              'canais': 48},

    #ResNet (§8.3): res15-narrow + hello_edge. 43.640 params (42,6 KB int8), 20.780.585 MACs,
    #acc val 94,24 ± 0,32% (baseline res8 95,31 ± 0,27%, −1,07 p.p.)
    'resnet': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,  #MFCC hello_edge: 10 x 49
               'canais': 19, 'n_blocos': 6, 'pool': None, 'dilatacao': True, 'conv_extra': True},   #res15-narrow

    #DS-CNN (§8.4, DS-CNN-S do Hello Edge): hello_edge + canais 48. 15.443 params (15,1 KB int8), 1.609.680 MACs,
    #acc val 92,40 ± 0,33% (baseline 94,08 ± 0,36%, −1,68 p.p.)
    'dscnn': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,   #MFCC hello_edge: 10 x 49
              'canais': 48, 'n_blocos': 4},

    #Atenção (§8.5, KWT: cada janela do MFCC vira um token): hello_edge + d_model 32. 41.251 params (40,3 KB int8),
    #1.860.000 MACs, acc val 92,22 ± 0,15% (baseline 40x101 + d_model 64 92,50 ± 1,37%, −0,28 p.p.)
    'atencao': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, #MFCC hello_edge: 10 x 49 (49 tokens)
                'd_model': 32, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0,
                'learning_rate': 0.001},     #Transformers costumam pedir lr menor
}


#Configurações de MFCC prontas. Uso: carregar_config("dscnn", **PRESETS_MFCC["hello_edge"])
PRESETS_MFCC = {
    #40 coeficientes x 101 janelas (janela 30 ms, passo 10 ms)
    "padrao": dict(n_mfcc=40, janela_ms=30, passo_ms=10, n_mels=40, center=True),
    #Igual ao artigo Hello Edge (e ao notebook do Luciano): 10 coeficientes x 49 janelas, entrada 8x menor
    "hello_edge": dict(n_mfcc=10, janela_ms=40, passo_ms=20, n_mels=40, center=False),
}

#Entrada SEM MFCC (§12 do relatório): o áudio cru entra como uma "matriz" 1 x 16000, no mesmo formato
#(B, n_mfcc, n_frames) do resto do pipeline. A normalização por amostra e um "MFCC aprendido" (FrenteOnda,
#modelos/onda.py) ficam DENTRO da rede. Uso: carregar_config("cnn1d", **PRESET_ONDA)
PRESET_ONDA = dict(entrada='onda', n_mfcc=1, normalizacao='nenhuma', spec_augment=False,
                   onda_frente='conv',      #'conv': "MFCC aprendido" | 'patches': pedaços de 10 ms crus (ver modelos/onda.py)
                   onda_filtros=40,         #canais do "MFCC aprendido" (o MFCC padrão tem 40 coeficientes)
                   onda_janela_ms=25,       #tamanho de cada filtro (400 amostras)
                   onda_passo_ms=10,        #passo entre janelas (160 amostras): cerca de 100 janelas por segundo
                   onda_center=False)       #True: preenche as bordas, como o MFCC com center=True


def sem_mfcc(hp):
    """Sobrescritas para treinar o modelo de `hp` SEM MFCC, com uma frente aprendida do MESMO formato do MFCC dele.

    Ex.: hello_edge (10 coef, janela 40 ms, passo 20 ms) -> 10 filtros de 40 ms com passo de 20 ms -> 10 x 49, a mesma
    entrada da rede. Assim a rede é idêntica e só o MFCC fixo é trocado por uma convolução treinada.
    Uso: config(nome, **sem_mfcc(config(nome)))
    """
    return dict(PRESET_ONDA, onda_frente='conv', onda_filtros=hp['n_mfcc'], onda_janela_ms=hp['janela_ms'],
                onda_passo_ms=hp['passo_ms'], onda_center=hp.get('center', True))


def carregar_config(modelo, mfcc=None, **sobrescritas):
    """O dicionário `hyperparameters` do modelo: HYPERPARAMETERS + HP_MODELOS[modelo] + sobrescritas.

    Qualquer chave pode ser sobrescrita na chamada, ex.: carregar_config("mlp", epocas=1, tarefa="gsc12").
    mfcc: None = o MFCC do próprio modelo (cada modelo pode ter o seu) | "padrao" ou "hello_edge" = força o preset.
    Sobrescritas com valor None são ignoradas. n_classes e n_frames são calculados a partir da tarefa e do MFCC.
    """
    if mfcc is not None:
        sobrescritas = {**PRESETS_MFCC[mfcc], **sobrescritas}
    #deepcopy: as listas (seeds, snrs_db) não podem ser compartilhadas com os dicionários do módulo
    return montar_hp(copy.deepcopy(HYPERPARAMETERS), copy.deepcopy(HP_MODELOS), modelo, **sobrescritas)


def montar_hp(hyperparameters, hp_modelos, nome, **sobrescritas):
    """Hiperparâmetros do modelo `nome` a partir de dicionários escritos no notebook (sem YAML).

    hyperparameters: o que é comum a todos os modelos | hp_modelos[nome]: só o que muda no modelo.
    Sobrescritas com valor None são ignoradas, e n_classes e n_frames
    são calculados a partir da tarefa e do MFCC.
    """
    hp = {**hyperparameters, **hp_modelos.get(nome, {}), "modelo": nome}
    for k, v in sobrescritas.items():
        if v is not None:
            hp[k] = v

    from dados.mfcc import calcular_n_frames
    from dados.speech_commands import classes_da_tarefa
    hp["n_classes"] = len(classes_da_tarefa(hp["tarefa"]))
    hp["n_frames"] = calcular_n_frames(hp)
    return hp


def rotulo_experimento(hp):
    """Nome curto da configuração, ex.: 'gsc12_10x49_por_amostra'. Separa runs e resultados de setups diferentes."""
    return f"{hp['tarefa']}_{hp['n_mfcc']}x{hp['n_frames']}_{hp.get('normalizacao', 'global')}"


def escolher_device():
    """GPU se houver (Colab Pro), senão CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pastas_projeto(raiz=None):
    """Pastas de dados, cache e runs (checkpoints, históricos e o resultados.csv).

    No Colab o setup define KWS_DRIVE (ex.: /content/drive/MyDrive/kws-borda) e KWS_DATA (/content/data):
    os dados brutos são lidos do disco local (rápido), e cache/runs ficam no Drive (persistem).
    Localmente, tudo fica dentro do próprio repositório.
    """
    raiz = Path(raiz or os.environ.get("KWS_DRIVE", RAIZ_REPO))
    pastas = {
        "dados": Path(os.environ.get("KWS_DATA", raiz / "data")),
        "cache": raiz / "cache",
        "runs": raiz / "runs",
        "resultados": raiz / "runs",      #resultados.csv fica junto dos runs
    }
    for p in pastas.values():
        p.mkdir(parents=True, exist_ok=True)
    return pastas
