"""Registro de resultados: cada experimento vira uma linha em runs/resultados.csv.

É a "tabela de experimentos" do grupo (mesma ideia do `registrar(...)` dos exercícios).
O notebook principal lê esse CSV para montar as tabelas e gráficos finais.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd


def registrar(caminho_csv, nome_modelo, seed, dataset="gsc_v2", **metricas):
    """Acrescenta uma linha ao CSV (cria o arquivo se não existir)."""
    linha = dict(data=datetime.now().strftime("%Y-%m-%d %H:%M"), modelo=nome_modelo, seed=seed,
                 dataset=dataset, **metricas)
    caminho_csv = Path(caminho_csv)
    df = pd.DataFrame([linha])
    if caminho_csv.exists():
        df = pd.concat([pd.read_csv(caminho_csv), df], ignore_index=True)
    df.to_csv(caminho_csv, index=False)
    print(f">>> registrado: {nome_modelo} (seed {seed}, {dataset}) | "
          + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metricas.items()))
    return linha


def carregar_resultados(caminho_csv):
    caminho_csv = Path(caminho_csv)
    return pd.read_csv(caminho_csv) if caminho_csv.exists() else pd.DataFrame()


def consolidar(df, metricas=("acc", "f1"), por=("modelo", "dataset")):
    """Média ± desvio entre as seeds: a "validação estatística" do projeto."""
    if df.empty:
        return df
    agg = df.groupby(list(por))[list(metricas)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{a}" for m, a in agg.columns]
    return agg.reset_index()
