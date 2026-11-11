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
    'normalizacao': 'global',   #'global': média/desvio por coeficiente medidos no TREINO | 'por_amostra': cada áudio pela sua própria média/desvio
    'spec_augment': True,       #máscara de tempo e de coeficientes, igual para todos
    'fracao': 1.0,              #fração do TREINO usada (1.0 = tudo). Validação e teste são sempre completos

    #Treino
    'epocas': 20,
    'batch_size': 128,
    'learning_rate': 0.003,
    'weight_decay': 0.01,
    'otimizador': 'adamw',
    'scheduler': 'onecycle',
    'label_smoothing': 0.1,
    'seeds': [0, 1, 2],         #3 rodadas independentes para média ± desvio
    'num_workers': 0,           #MFCC já está em memória: processos extras só adicionam overhead

    #Avaliação
    'latencia_repeticoes': 300,
    'latencia_aquecimento': 30,
    'latencia_threads': 1,      #simula um dispositivo restrito (1 núcleo)
    'snrs_db': [20, 10, 5, 0, -5],
    'margem_pp': 2.0,           #critério do projeto: um modelo menor "serve" se ficar até 2 p.p. abaixo do baseline (validação)
}


#Só o que MUDA em cada modelo
HP_MODELOS = {
    #MLP — baseline: achata o MFCC e passa por camadas densas.
    #Versão do Luciano (notebooks/experimentos/Projeto_MLP.ipynb, 79,6% val em gsc12 + MFCC hello_edge)
    'mlp': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,       #MFCC hello_edge: 10 x 49
            'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False},

    #CNN 1D — vencedora da varredura (playground_cnn1d_guilherme.ipynb, 13 configurações x 3 seeds, GSC completo):
    #o MENOR modelo a menos de 2 p.p. (validação) do baseline 40x101 + 64 canais.
    #hello_edge + canais 48: 24.275 params (23,7 KB int8), 383.280 MACs, acc val 92,46 ± 0,13% (baseline 93,51 ± 0,06%)
    'cnn1d': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,     #MFCC hello_edge: 10 x 49
              'canais': 48},

    #ResNet — vencedora da varredura (playground_resnet_guilherme.ipynb, 15 configurações x 3 seeds, GSC completo):
    #o MENOR modelo a menos de 2 p.p. (validação) do baseline res8 40x101.
    #res15-narrow + hello_edge: 43.640 params (42,6 KB int8), 20.780.585 MACs, acc val 94,88 ± 0,09% (baseline 95,82 ± 0,03%)
    'resnet': {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False,    #MFCC hello_edge: 10 x 49
               'canais': 19, 'n_blocos': 6, 'pool': None, 'dilatacao': True, 'conv_extra': True},

    #DS-CNN (Hello Edge, Zhang et al. 2017) — convoluções separáveis em profundidade
    'dscnn': {
        'n_mfcc': 10,
        'janela_ms': 40,
        'passo_ms': 20,
        'center': False,
        'canais': 64,
        'n_blocos': 4
    },

    #Atenção — KWT-tiny (Berg et al. 2021): cada frame do MFCC vira um token
    'atencao': {'d_model': 64, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0,
                'learning_rate': 0.001},     #Transformers costumam pedir lr menor
}


#Configurações de MFCC prontas. Uso: carregar_config("dscnn", **PRESETS_MFCC["hello_edge"])
PRESETS_MFCC = {
    #40 coeficientes x 101 janelas (janela 30 ms, passo 10 ms)
    "padrao": dict(n_mfcc=40, janela_ms=30, passo_ms=10, n_mels=40, center=True),
    #Igual ao artigo Hello Edge (e ao notebook do Luciano): 10 coeficientes x 49 janelas, entrada 8x menor
    "hello_edge": dict(n_mfcc=10, janela_ms=40, passo_ms=20, n_mels=40, center=False),
}


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
