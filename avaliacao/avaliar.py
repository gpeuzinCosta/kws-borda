"""Custo computacional e latência (MFCC × rede) de todos os modelos implementados.

    python avaliacao/avaliar.py                  # todos os implementados
    python avaliacao/avaliar.py --modelos mlp dscnn --librosa

Não precisa de pesos treinados: parâmetros, MACs e latência dependem só da arquitetura.
"""
import sys
from pathlib import Path

#Permite rodar direto (python avaliacao/avaliar.py) a partir da raiz do repositório
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from configs import PRESETS_MFCC, carregar_config
from dados import ExtratorMFCC
from avaliacao import custo, medir_latencia
from modelos import criar_modelo, esta_implementado, listar_modelos

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", default=[m["nome"] for m in listar_modelos()])
    ap.add_argument("--librosa", action="store_true", help="mede também o MFCC com librosa")
    ap.add_argument("--repeticoes", type=int, default=300)
    ap.add_argument("--mfcc", choices=list(PRESETS_MFCC), help="preset de MFCC (padrão: o do base.yaml)")
    args = ap.parse_args()

    linhas = []
    for nome in args.modelos:
        if not esta_implementado(nome):
            print(f"[pulando] {nome}: ainda não implementado")
            continue
        hp = carregar_config(nome, **PRESETS_MFCC.get(args.mfcc, {}))
        model = criar_modelo(nome, hp)
        linhas.append(dict(modelo=nome, **custo(model),
                           **medir_latencia(model, ExtratorMFCC.de_hparams(hp), threads=hp["latencia_threads"],
                                            repeticoes=args.repeticoes, com_librosa=args.librosa)))
    print(pd.DataFrame(linhas).round(3).to_string(index=False))
