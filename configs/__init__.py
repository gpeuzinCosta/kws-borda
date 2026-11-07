"""configs — hiperparâmetros (YAML), pastas do projeto e device.

Os YAMLs são organizados em seções (dados, treino, avaliacao, modelo) para ficarem legíveis,
mas o código usa um dicionário "achatado" `hyperparameters`, no mesmo estilo dos exercícios:

    hp = carregar_config("dscnn")
    hp["batch_size"], hp["learning_rate"], hp["canais"] ...
"""
from pathlib import Path

import os

import yaml

#configs/__init__.py -> a pasta configs/ e a raiz do repositório
PASTA_CONFIGS = Path(__file__).resolve().parent
RAIZ_REPO = PASTA_CONFIGS.parent


def _mesclar(base, novo):
    """Mescla dicionários aninhados: o que está em `novo` sobrescreve `base`."""
    out = dict(base)
    for k, v in novo.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _mesclar(out[k], v)
        else:
            out[k] = v
    return out


def _achatar(cfg):
    """{'treino': {'batch_size': 128}, 'modelo': {'nome': 'mlp'}} -> {'batch_size': 128, 'modelo': 'mlp'}"""
    plano = {}
    for secao, valores in cfg.items():
        if not isinstance(valores, dict):
            plano[secao] = valores
            continue
        for k, v in valores.items():
            #O nome do modelo vira a chave 'modelo' para ficar mais legível
            chave = "modelo" if (secao == "modelo" and k == "nome") else k
            if chave in plano:
                raise KeyError(f"Hiperparâmetro '{chave}' repetido em mais de uma seção do YAML")
            plano[chave] = v
    return plano


#Configurações de MFCC prontas. Uso: carregar_config("dscnn", **PRESETS_MFCC["hello_edge"])
PRESETS_MFCC = {
    #40 coeficientes x 101 janelas (janela 30 ms, passo 10 ms)
    "padrao": dict(n_mfcc=40, janela_ms=30, passo_ms=10, n_mels=40, center=True),
    #Igual ao artigo Hello Edge (e ao notebook do Luciano): 10 coeficientes x 49 janelas, entrada 8x menor
    "hello_edge": dict(n_mfcc=10, janela_ms=40, passo_ms=20, n_mels=40, center=False),
}


def carregar_config(modelo, pasta=PASTA_CONFIGS, mfcc=None, **sobrescritas):
    """Carrega configs/base.yaml + configs/<modelo>.yaml e devolve o dicionário `hyperparameters`.

    Qualquer chave pode ser sobrescrita na chamada, ex.: carregar_config("mlp", epocas=1, tarefa="gsc12").
    mfcc: None = o MFCC do YAML do modelo (cada modelo pode ter o seu) | "padrao" ou "hello_edge" = força o preset.
    Sobrescritas com valor None são ignoradas. n_classes e n_frames são calculados a partir da tarefa e do MFCC.
    """
    pasta = Path(pasta)
    base = yaml.safe_load(open(pasta / "base.yaml"))
    especifico = yaml.safe_load(open(pasta / f"{modelo}.yaml")) or {}
    hp = _achatar(_mesclar(base, especifico))
    if mfcc is not None:
        hp.update(PRESETS_MFCC[mfcc])
    for k, v in sobrescritas.items():
        if v is not None:
            hp[k] = v

    from dados.mfcc import calcular_n_frames
    from dados.speech_commands import classes_da_tarefa
    hp["n_classes"] = len(classes_da_tarefa(hp["tarefa"]))
    hp["n_frames"] = calcular_n_frames(hp)
    return hp


def montar_hp(hyperparameters, hp_modelos, nome, **sobrescritas):
    """Hiperparâmetros do modelo `nome` a partir de dicionários escritos no notebook (sem YAML).

    hyperparameters: o que é comum a todos os modelos | hp_modelos[nome]: só o que muda no modelo.
    Mesmas regras do carregar_config: sobrescritas com valor None são ignoradas, e n_classes e n_frames
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
