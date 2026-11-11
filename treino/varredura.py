"""Varredura de configurações de UM modelo: quanto dá para encolher sem perder reconhecimento?

Trazido dos capítulos 8 e 9 do playground da CNN 1D (notebooks/experimentos/playground_cnn1d_guilherme.ipynb).
Serve para qualquer modelo do registro: a rede é criada por `criar_modelo(nome_modelo, hp)`, e as configurações
são uma lista de (nome, o que muda em relação ao hp base): CONFIGS_CNN1D, CONFIGS_RESNET, CONFIGS_MLP,
CONFIGS_DSCNN e CONFIGS_ATENCAO.

Critério do projeto (definido ANTES de ver os resultados, medido na VALIDAÇÃO):
o vencedor é o MENOR modelo (parâmetros, depois MACs) até `margem_pp` p.p. abaixo do baseline.

Uso (o mesmo no relatório e nos playgrounds):
    tabela, modelos = rodar_varredura("cnn1d", CONFIGS_CNN1D, hp, seeds, raiz_gsc, pastas)
    latencias = medir_latencias_varredura("cnn1d", CONFIGS_CNN1D, hp)
    tabela = anexar_latencias(tabela, latencias)
    resumo = montar_resumo(tabela, hp["fracao"], hp["margem_pp"])
    vencedor = escolher_vencedor(resumo)
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm as tqdm_texto    #barra em TEXTO: a do tqdm.auto é um widget, que não aparece no VS Code com o Colab

from avaliacao.custo import custo
from avaliacao.graficos import plot_matriz_confusao
from avaliacao.latencia import medir_latencia
from avaliacao.metricas import acuracia, f1_macro, matriz_confusao
from configs import escolher_device
from dados.mfcc import calcular_n_frames
from dados.speech_commands import carregar_gsc, classes_da_tarefa, criar_dataloaders, nome_cache
from modelos import criar_modelo
from treino.loop import carregar_run, mesma_politica, prever, treinar

#As configurações da varredura da CNN 1D (mesma lista do playground)
CONFIGS_CNN1D = [
    ('baseline',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64}),
    ('sem spec_augment',         {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'spec_augment': False}),
    ('norm por_amostra',         {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'normalizacao': 'por_amostra'}),
    ('sem weight_decay',         {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'weight_decay': 0.0}),
    ('canais 32',                {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 32}),
    ('canais 16',                {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 16}),
    ('canais 8',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 8}),
    ('canais 4',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 4}),
    ('entrada hello_edge',       {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 64}),
    ('hello_edge + canais 16',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 16}),
    ('hello_edge + canais 8',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 8}),
    ('hello_edge + canais 48',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 48}),
    ('hello_edge + canais 32',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 32}),
]
#No playground o baseline era o próprio dicionário (40x101, 64 canais). Aqui cada configuração diz o MFCC e os
#canais explicitamente, porque o HP_MODELOS['cnn1d'] (configs/__init__.py) guarda o VENCEDOR (10x49, 48 canais), e não o baseline

#As configurações da varredura da ResNet (mesmos nomes do playground_resnet_guilherme.ipynb, para reaproveitar os
#treinos salvos no Drive). As famílias são as do Tang & Lin (2018); 'pool' é lista (não tupla) porque o config.yaml
#de cada treino não guarda tuplas. MFCC e arquitetura explícitos em cada linha: o HP_MODELOS['resnet'] guarda o
#VENCEDOR (res15-narrow + hello_edge), e não o baseline (res8 com o MFCC padrão)
CONFIGS_RESNET = [
    ('baseline',                    {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 45, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('sem spec_augment',            {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 45, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False, 'spec_augment': False}),
    ('norm por_amostra',            {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 45, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False, 'normalizacao': 'por_amostra'}),
    ('sem weight_decay',            {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 45, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False, 'weight_decay': 0.0}),
    ('res8-narrow',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 19, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('res15-narrow',                {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 19, 'n_blocos': 6, 'pool': None, 'dilatacao': True, 'conv_extra': True}),
    ('res26-narrow',                {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 19, 'n_blocos': 12, 'pool': [2, 2], 'dilatacao': False, 'conv_extra': False}),
    ('res8 canais 12',              {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 12, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('res8 canais 8',               {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 8, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('res8 + hello_edge',           {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 45, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('res8-narrow + hello_edge',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 19, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('res15-narrow + hello_edge',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 19, 'n_blocos': 6, 'pool': None, 'dilatacao': True, 'conv_extra': True}),
    ('res26-narrow + hello_edge',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 19, 'n_blocos': 12, 'pool': [2, 2], 'dilatacao': False, 'conv_extra': False}),
    ('res8 canais 12 + hello_edge', {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 12, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
    ('res8 canais 8 + hello_edge',  {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 8, 'n_blocos': 3, 'pool': [3, 4], 'dilatacao': False, 'conv_extra': False}),
]

#As varreduras de MLP, DS-CNN e Atenção rodam inteiras no notebook principal (§8.1, §8.4 e §8.5). Mesmo desenho das
#anteriores: o baseline é a arquitetura de partida com o MFCC padrão (40x101), depois as 3 variações de treino,
#modelos menores e a entrada hello_edge (10x49). MFCC e arquitetura explícitos em cada linha

#MLP: arquitetura de partida = a do Projeto_MLP.ipynb (3 camadas de 128, dropout 0.1, sem BatchNorm)
CONFIGS_MLP = [
    ('baseline',                  {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False}),
    ('sem spec_augment',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False, 'spec_augment': False}),
    ('norm por_amostra',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False, 'normalizacao': 'por_amostra'}),
    ('sem weight_decay',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False, 'weight_decay': 0.0}),
    ('hidden 64',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 64, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False}),
    ('hidden 32',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 32, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False}),
    ('2 camadas',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 128, 'n_camadas': 2, 'dropout': 0.1, 'batchnorm': False}),
    ('com batchnorm',             {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': True}),
    ('entrada hello_edge',        {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'hidden': 128, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False}),
    ('hello_edge + hidden 64',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'hidden': 64, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False}),
    ('hello_edge + hidden 32',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'hidden': 32, 'n_camadas': 3, 'dropout': 0.1, 'batchnorm': False}),
    ('hello_edge + 2 camadas',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'hidden': 128, 'n_camadas': 2, 'dropout': 0.1, 'batchnorm': False}),
]

#DS-CNN: arquitetura de partida = DS-CNN-S do Hello Edge (64 canais, 4 blocos separáveis)
CONFIGS_DSCNN = [
    ('baseline',                  {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'n_blocos': 4}),
    ('sem spec_augment',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'n_blocos': 4, 'spec_augment': False}),
    ('norm por_amostra',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'n_blocos': 4, 'normalizacao': 'por_amostra'}),
    ('sem weight_decay',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'n_blocos': 4, 'weight_decay': 0.0}),
    ('canais 32',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 32, 'n_blocos': 4}),
    ('canais 16',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 16, 'n_blocos': 4}),
    ('2 blocos',                  {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'canais': 64, 'n_blocos': 2}),
    ('entrada hello_edge',        {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 64, 'n_blocos': 4}),
    ('hello_edge + canais 48',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 48, 'n_blocos': 4}),
    ('hello_edge + canais 32',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 32, 'n_blocos': 4}),
    ('hello_edge + canais 16',    {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 16, 'n_blocos': 4}),
    ('hello_edge + 2 blocos',     {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'canais': 64, 'n_blocos': 2}),
]

#Atenção: arquitetura de partida = KWT com d_model 64, 3 camadas, 1 cabeça e sem dropout (o lr 1e-3 vem do HP_MODELOS)
CONFIGS_ATENCAO = [
    ('baseline',                  {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 64, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0}),
    ('sem spec_augment',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 64, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0, 'spec_augment': False}),
    ('norm por_amostra',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 64, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0, 'normalizacao': 'por_amostra'}),
    ('sem weight_decay',          {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 64, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0, 'weight_decay': 0.0}),
    ('d_model 32',                {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 32, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0}),
    ('d_model 16',                {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 16, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0}),
    ('2 camadas',                 {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 64, 'n_camadas': 2, 'n_heads': 1, 'dropout': 0}),
    ('1 camada',                  {'n_mfcc': 40, 'janela_ms': 30, 'passo_ms': 10, 'center': True, 'd_model': 64, 'n_camadas': 1, 'n_heads': 1, 'dropout': 0}),
    ('entrada hello_edge',        {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'd_model': 64, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0}),
    ('hello_edge + d_model 32',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'd_model': 32, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0}),
    ('hello_edge + d_model 16',   {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'd_model': 16, 'n_camadas': 3, 'n_heads': 1, 'dropout': 0}),
    ('hello_edge + 1 camada',     {'n_mfcc': 10, 'janela_ms': 40, 'passo_ms': 20, 'center': False, 'd_model': 64, 'n_camadas': 1, 'n_heads': 1, 'dropout': 0}),
]


def aplicar(hp, mudancas, **extra):
    """hp base + as mudanças de uma configuração. Recalcula n_frames (muda com o MFCC)."""
    novo = {**hp, **mudancas, **extra}
    novo['n_frames'] = calcular_n_frames(novo)
    return novo


mfcc_em_memoria = {'nome': None, 'dados': None}


def carregar_dados(hp, raiz, pasta_cache):
    """MFCC de treino/val/teste para o MFCC de `hp` (com cache no disco).
    Só recarrega se a configuração do MFCC mudou desde o pedido anterior: a varredura alterna pouco."""
    nome = nome_cache(hp)

    #Só recarrega se a configuração do MFCC mudou desde o experimento anterior
    if mfcc_em_memoria['nome'] != nome:
        #Solta o MFCC antigo ANTES de carregar o novo
        mfcc_em_memoria['dados'] = None

        mfcc_em_memoria['dados'] = carregar_gsc(hp, raiz, pasta_cache)
        mfcc_em_memoria['nome'] = nome

    return mfcc_em_memoria['dados']


def hp_arquitetura(nome_modelo, hp):
    """Os hiperparâmetros de `hp` que o __init__ do modelo aceita (ex.: canais, n_blocos)."""
    modelo = criar_modelo(nome_modelo, hp)
    aceitos = inspect.signature(type(modelo).__init__).parameters
    return {k: hp[k] for k in aceitos if k in hp and k not in ('self', 'n_classes', 'n_mfcc', 'n_frames')}


def registrar_linha(tabela, nome, seed, hp, hist, val, teste):
    """Acrescenta uma linha à tabela de resultados (uma por treino)."""

    #Custo medido num modelo NOVO: o custo() roda um forward em modo train(), e isso alteraria as
    #estatísticas acumuladas do BatchNorm do modelo treinado. Parâmetros e MACs não dependem dos pesos
    c = custo(criar_modelo(hp['modelo'], hp))

    tabela.append({
        'modelo': hp['modelo'], 'experimento': nome, 'seed': seed, 'fracao': hp['fracao'], 'epocas': hp['epocas'],
        'spec_augment': hp['spec_augment'], 'normalizacao': hp['normalizacao'],
        'weight_decay': hp['weight_decay'], **hp_arquitetura(hp['modelo'], hp),
        'entrada': f"{hp['n_mfcc']}x{hp['n_frames']}",
        'params': c['params'], 'kb_int8': round(c['kb_int8'], 1), 'macs': c['macs'],
        'acc_val': hist['melhor_acc'],            #melhor acurácia de VALIDAÇÃO (a que seleciona)
        'f1_val': val['f1'],
        'melhor_epoca': hist['melhor_epoca'], 'paciencia': hp.get('paciencia'),
        'epocas_treinadas': hist.get('epocas_treinadas', len(hist['acc'])), 'gap': hist['gap'], 'tempo_min': hist['tempo_min'],
        'acc_teste': teste['acc'], 'f1_teste': teste['f1'],    #só para reportar: nada é escolhido com elas
    })


def _avaliar(model, loader):
    y_pred, y_true = prever(model, loader)
    return dict(acc=acuracia(y_pred, y_true), f1=f1_macro(y_pred, y_true))


def rodar_varredura(nome_modelo, configs, hp, seeds, raiz, pastas, retreinar=False, device=None):
    """Treina cada configuração com cada seed. Devolve (tabela: DataFrame, modelos: {(experimento, seed): model}).

    - A tabela fica salva em <pastas['resultados']>/varredura_<modelo>.csv e os pesos em <pastas['runs']>.
      Com retreinar=False, o que já foi treinado (mesma configuração, seed, fração e épocas) é CARREGADO.
    - hp: o dicionário base (ex.: carregar_config('cnn1d')); a fração do treino é hp['fracao'].
    """
    device = device or escolher_device()
    fracao = hp['fracao']
    caminho_csv = Path(pastas['resultados']) / f'varredura_{nome_modelo}.csv'

    tabela = []
    if caminho_csv.exists() and not retreinar:
        tabela = pd.read_csv(caminho_csv).to_dict('records')
    ja_feitos = {(r['experimento'], r['seed'], r['fracao'], r['epocas']) for r in tabela}
    modelos = {}

    #Seed por fora: se algo falhar no meio, o que já terminou é uma comparação COMPLETA com a seed 0
    for seed in seeds:
        barra = tqdm_texto(configs, desc=f'{nome_modelo} | seed {seed}')
        for nome, mudancas in barra:
            barra.set_postfix_str(nome)
            hp_exp = aplicar(hp, mudancas, modelo=nome_modelo, seed=seed, fracao=fracao)

            loaders, _ = criar_dataloaders(hp_exp, carregar_dados(hp_exp, raiz, pastas['cache']))
            model = criar_modelo(nome_modelo, hp_exp)

            #Já treinado antes (no Drive)? Só serve se for da mesma fração, teto de épocas, scheduler e early stop
            run = None
            if not retreinar:
                run = carregar_run(pastas['runs'], hp_exp, seed, nome_run=nome)
                if run is not None and not mesma_politica(run[2], hp_exp):
                    run = None

            if run is not None:
                model.load_state_dict(run[0])
                model = model.to(device)
                hist = run[1]
                barra.write(f'{nome} (seed {seed}): carregado de {hist.get("pasta")}')    #write em vez de print: não quebra a barra
            else:
                model, hist = treinar(model, loaders, hp_exp, seed=seed, pasta_runs=pastas['runs'],
                                      device=device, verbose=False, nome_run=nome)
                barra.write(f'{nome:30s} | {sum(p.numel() for p in model.parameters()):7d} params | '
                            f'melhor acc val={hist["melhor_acc"]*100:.2f}% (época {hist["melhor_epoca"]:2d}) | '
                            f'{hist["tempo_min"]:.1f} min')

            #Teste medido logo depois do treino. Registra só o que ainda não está na tabela
            if (nome, seed, fracao, hp_exp['epocas']) not in ja_feitos:
                registrar_linha(tabela, nome, seed, hp_exp, hist, _avaliar(model, loaders['val']),
                                _avaliar(model, loaders['teste']))
                ja_feitos.add((nome, seed, fracao, hp_exp['epocas']))
                pd.DataFrame(tabela).to_csv(caminho_csv, index=False)    #salva a cada treino: sobrevive a uma queda

            modelos[(nome, seed)] = model
            del loaders    #libera as cópias normalizadas dos dados

    #Devolve só as configurações PEDIDAS: o CSV no Drive pode ter outras (ex.: o relatório roda só o baseline e o
    #vencedor, e o playground deixou as 15 configurações salvas)
    tabela = pd.DataFrame(tabela)
    nomes = [nome for nome, _ in configs]
    pedidas = (tabela['fracao'] == fracao) & (tabela['epocas'] == hp['epocas']) & tabela['experimento'].isin(nomes)
    return tabela[pedidas].reset_index(drop=True), modelos


def medir_latencias_varredura(nome_modelo, configs, hp, passadas=3):
    """Uma medida de latência por CONFIGURAÇÃO (a latência não depende da seed nem dos pesos).
    Devolve {experimento: dicionário da medir_latencia}."""

    #Aquecimento descartado: no piloto, a primeira medida depois da varredura saiu inflada
    #(baseline: MFCC 0,83 ms contra ~0,67 ms nas configurações com o mesmo MFCC)
    hp_aq = aplicar(hp, configs[0][1], modelo=nome_modelo)
    _ = medir_latencia(criar_modelo(nome_modelo, hp_aq), hp_aq, torch.zeros(hp_aq['n_mfcc'], 1),
                       torch.ones(hp_aq['n_mfcc'], 1), rodadas=2)

    medidas = {}    #experimento -> uma medida por passada
    ordem = list(configs)

    #3 passadas, a segunda e a terceira na ordem INVERSA: se a máquina ficar mais lenta (ou mais rápida) com o tempo,
    #a deriva pesa de um lado numa passada e do outro na seguinte, e a média das duas compensa
    for passada in range(passadas):
        for nome, mudancas in ordem:
            hp_exp = aplicar(hp, mudancas, modelo=nome_modelo)

            if hp_exp['normalizacao'] == 'global':
                media = torch.zeros(hp_exp['n_mfcc'], 1)    #o tempo não depende dos valores, só do formato
                desvio = torch.ones(hp_exp['n_mfcc'], 1)
            else:
                media = desvio = None                       #por_amostra: calcula média e desvio de cada áudio

            model_lat = criar_modelo(nome_modelo, hp_exp)

            if nome not in medidas:
                medidas[nome] = []
            medidas[nome].append(medir_latencia(model_lat, hp_exp, media, desvio))
        ordem = ordem[::-1]

    latencias = {}
    for nome, lista in medidas.items():
        latencias[nome] = dict(lista[0])                       #cpu e os outros campos

        #MÍNIMO entre as passadas (e não a média): ruído de máquina compartilhada só deixa a medida MAIS LENTA,
        #nunca mais rápida. Então a passada mais rápida é a menos contaminada
        for coluna in ['mfcc_ms', 'norm_ms', 'rede_ms']:
            latencias[nome][coluna] = float(np.min([m[coluna] for m in lista]))
        latencias[nome]['preproc_ms'] = latencias[nome]['mfcc_ms'] + latencias[nome]['norm_ms']
        latencias[nome]['preproc_sobre_rede'] = latencias[nome]['preproc_ms'] / latencias[nome]['rede_ms']
        latencias[nome]['fator_x8'] = float(np.median([m['fator_x8'] for m in lista]))
        latencias[nome]['estavel'] = all(m['estavel'] for m in lista)

        l = latencias[nome]
        passadas_txt = ' / '.join([f"{m['rede_ms']:.3f}" for m in lista])
        print(f"{nome:26s} | MFCC {l['mfcc_ms']:.3f} ms | norm {l['norm_ms']:.3f} ms | rede {l['rede_ms']:.3f} ms | "
              f"preproc/rede {l['preproc_sobre_rede']:.2f}x | estável: {l['estavel']} | rede nas passadas: {passadas_txt}")

    #Todas as latências precisam sair da MESMA máquina: o Colab muda de CPU entre sessões
    cpus = {l['cpu'] for l in latencias.values()}
    if len(cpus) > 1:
        print('AVISO: as medidas vieram de CPUs diferentes:', cpus)
    else:
        print('CPU (todas as medidas):', cpus.pop())
    return latencias


def anexar_latencias(tabela, latencias):
    """Anexa a latência a TODAS as linhas da tabela (a latência é da configuração, igual para todas as seeds)."""
    tabela = tabela.copy()
    colunas_lat = ['mfcc_ms', 'norm_ms', 'rede_ms', 'preproc_ms', 'preproc_sobre_rede', 'fator_x8', 'estavel', 'cpu']
    for coluna in colunas_lat:
        tabela[coluna] = [latencias[nome][coluna] for nome in tabela['experimento']]
    return tabela


def montar_resumo(tabela, fracao, margem_pp, referencia='baseline'):
    """Média ± desvio entre as seeds, por experimento, e o critério dos `margem_pp` p.p. (na validação)."""
    df = tabela
    if 'mfcc_ms' not in df.columns:
        raise RuntimeError('Rode medir_latencias_varredura + anexar_latencias antes do resumo')
    df = df[df['fracao'] == fracao]

    resumo = df.groupby('experimento', sort=False).agg(
        runs=('seed', 'count'),
        acc_val=('acc_val', 'mean'), acc_val_dp=('acc_val', 'std'),
        acc_teste=('acc_teste', 'mean'), acc_teste_dp=('acc_teste', 'std'),
        f1_teste=('f1_teste', 'mean'), f1_teste_dp=('f1_teste', 'std'),
        entrada=('entrada', 'first'), params=('params', 'first'), kb_int8=('kb_int8', 'first'), macs=('macs', 'first'),
        mfcc_ms=('mfcc_ms', 'first'), norm_ms=('norm_ms', 'first'), rede_ms=('rede_ms', 'first'),
        preproc_sobre_rede=('preproc_sobre_rede', 'first'), fator_x8=('fator_x8', 'first'),
        tempo_min=('tempo_min', 'mean'))

    for coluna in ['acc_val', 'acc_val_dp', 'acc_teste', 'acc_teste_dp', 'f1_teste', 'f1_teste_dp']:
        resumo[coluna] = resumo[coluna] * 100     #em %. O desvio fica NaN com 1 seed: normal

    #Critério definido ANTES de ver os resultados, medido na VALIDAÇÃO
    resumo['dif_baseline'] = resumo['acc_val'] - resumo.loc[referencia, 'acc_val']
    resumo['dentro_2pp'] = resumo['dif_baseline'] >= -margem_pp
    return resumo


def formatar(media, dp):
    if pd.isna(dp):
        return f'{media:.2f}'
    return f'{media:.2f} ± {dp:.2f}'


def tabela_slide(resumo):
    """Resumo por experimento (média ± desvio entre as seeds): a tabela dos slides."""
    return pd.DataFrame({
        'Entrada': resumo['entrada'],
        'Params': resumo['params'],
        'KB int8': resumo['kb_int8'],
        'MACs': resumo['macs'],
        'Acc val (%)': [formatar(m, d) for m, d in zip(resumo['acc_val'], resumo['acc_val_dp'])],
        'Acc teste (%)': [formatar(m, d) for m, d in zip(resumo['acc_teste'], resumo['acc_teste_dp'])],
        'F1 teste (%)': [formatar(m, d) for m, d in zip(resumo['f1_teste'], resumo['f1_teste_dp'])],
        'MFCC (ms)': resumo['mfcc_ms'].round(3),
        'Rede (ms)': resumo['rede_ms'].round(3),
        'Pré-proc / rede': resumo['preproc_sobre_rede'].round(2),
        'Dif. val vs baseline (p.p.)': resumo['dif_baseline'].round(2),
        'Dentro de 2 p.p.': resumo['dentro_2pp'],
    })


def escolher_vencedor(resumo):
    """O MENOR modelo (parâmetros, depois MACs) dentro do critério. Devolve o nome do experimento."""
    aprovados = resumo[resumo['dentro_2pp']].sort_values(['params', 'macs'])
    print(f"Menor modelo dentro de 2 p.p. do baseline: {aprovados.index[0]} "
          f"({aprovados.iloc[0]['params']:,.0f} parâmetros, {aprovados.iloc[0]['macs']:,.0f} MACs)")
    if resumo['runs'].max() < 3:
        print('ATENÇÃO: com menos de 3 seeds isto é só um palpite, não uma conclusão')
    return aprovados.index[0]


def matrizes_escolhidos(resumo, modelos, configs, hp, seeds, raiz, pasta_cache, referencia='baseline'):
    """Matrizes de confusão (no TESTE) de 3 configurações, escolhidas pela VALIDAÇÃO e pelo critério do projeto:
    1) o baseline | 2) o MENOR modelo dentro de 2 p.p. | 3) o que ficou logo abaixo da linha (mostra o que quebra primeiro)
    """
    aprovados = resumo[resumo['dentro_2pp']].sort_values(['params', 'macs'])
    reprovados = resumo[~resumo['dentro_2pp']].sort_values('acc_val', ascending=False)

    escolhidos = [(referencia, 'referência')]
    if aprovados.index[0] != referencia:
        escolhidos.append((aprovados.index[0], 'menor modelo dentro de 2 p.p.'))
    if len(reprovados) > 0:
        escolhidos.append((reprovados.index[0], 'logo abaixo da linha dos 2 p.p.'))

    mudancas_por_nome = dict(configs)
    classes = classes_da_tarefa(hp['tarefa'])

    for nome, papel in escolhidos:
        hp_exp = aplicar(hp, mudancas_por_nome[nome], seed=seeds[0])

        #O DataModule do treino foi apagado no laço (del dm): recria só para ter o teste com a MESMA normalização.
        #A normalização não depende da seed do treino, então um DataModule serve para todas as seeds
        loaders, _ = criar_dataloaders(hp_exp, carregar_dados(hp_exp, raiz, pasta_cache))

        #As contagens de cada seed SOMAM: a matriz fica com 3x mais exemplos e não depende da sorte de uma seed só
        cm = np.zeros((len(classes), len(classes)))
        acc_teste = []
        for seed in seeds:
            y_pred, y_true = prever(modelos[(nome, seed)], loaders['teste'])
            cm = cm + matriz_confusao(y_pred, y_true, len(classes), normalizar=False)
            acc_teste.append(acuracia(y_pred, y_true))
        del loaders

        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)   #normalizada por linha

        #O acc_val do resumo já está em % (a montar_resumo multiplica por 100)
        plot_matriz_confusao(cm_norm, classes,
                             titulo=f"{nome} ({papel}) | {resumo.loc[nome, 'params']:,.0f} params | "
                                    f"acc val {resumo.loc[nome, 'acc_val']:.2f}% | acc teste {np.mean(acc_teste)*100:.2f}%")
