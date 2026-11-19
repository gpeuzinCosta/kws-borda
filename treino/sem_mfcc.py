"""Treino SEM MFCC (§11 do relatório): ajudantes para comparar o mesmo modelo com e sem MFCC.

A entrada sem MFCC e a frente aprendida estão em dados/mfcc.py (ExtratorOnda) e modelos/onda.py; as
sobrescritas de hiperparâmetros, em configs.sem_mfcc.
"""
import numpy as np

from avaliacao.custo import custo
from avaliacao.metricas import acuracia
from modelos import criar_modelo
from treino.loop import carregar_run, mesma_politica, prever, treinar


def treinar_ou_carregar(nome, hp, loaders, seed, pasta_runs, device, retreinar=False, nome_run=None):
    """O mesmo caminho da §5: carrega o treino salvo no Drive (mesma política) ou treina. Devolve (model, hist)."""
    model = criar_modelo(nome, hp)
    run = None if retreinar else carregar_run(pasta_runs, hp, seed, nome_run=nome_run)
    if run is not None and mesma_politica(run[2], hp):
        model.load_state_dict(run[0])
        hist = run[1]
        print(f"{nome} | seed {seed}: carregado do Drive ({hist.get('pasta')}) | melhor acc val {max(hist['acc'])*100:.2f}%")
        return model.to(device), hist
    return treinar(model, loaders, hp, seed=seed, pasta_runs=pasta_runs, device=device, nome_run=nome_run)


def resumo_treino(rotulo, nome, hp, hist, acc_teste):
    """Uma linha da comparação com x sem MFCC: como o treino andou, o resultado e o custo da rede."""
    c = custo(criar_modelo(nome, hp))
    return {"modelo": nome, "entrada": rotulo, "formato": f"{hp['n_mfcc']}x{hp['n_frames']}",
            "melhor acc val (%)": hist["melhor_acc"] * 100,
            "acc teste (%)": acc_teste * 100,
            "melhor época": hist["melhor_epoca"],
            "épocas treinadas": hist.get("epocas_treinadas", len(hist["acc"])),
            "acc val na época 5 (%)": hist["acc"][min(4, len(hist["acc"]) - 1)] * 100,     #quem aprende mais rápido
            "s por época": float(np.mean(hist["tempo_s"])) if hist.get("tempo_s") else None,
            "tempo total (min)": hist.get("tempo_min"),
            "gap (perda val - treino)": hist.get("gap"),
            "params": c["params"], "macs": c["macs"]}


def acuracia_teste(model, loader):
    y_pred, y_true = prever(model, loader)
    return acuracia(y_pred, y_true)
