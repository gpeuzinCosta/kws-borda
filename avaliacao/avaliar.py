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
import torch

from configs import PRESETS_MFCC, carregar_config
from avaliacao import custo, medir_latencia
from modelos import criar_modelo, esta_implementado, listar_modelos

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", default=[m["nome"] for m in listar_modelos()])
    ap.add_argument("--librosa", action="store_true", help="mede também o MFCC com librosa")
    ap.add_argument("--repeticoes", type=int, default=300)
    ap.add_argument("--mfcc", choices=list(PRESETS_MFCC), help="preset de MFCC (padrão: o MFCC do próprio modelo)")
    args = ap.parse_args()

    linhas = []
    for nome in args.modelos:
        if not esta_implementado(nome):
            print(f"[pulando] {nome}: ainda não implementado")
            continue
        hp = carregar_config(nome, **PRESETS_MFCC.get(args.mfcc, {}))
        model = criar_modelo(nome, hp)
        #Sem pesos treinados: média 0 e desvio 1 têm o mesmo custo (o tempo depende só do formato)
        if hp["normalizacao"] == "global":
            media, desvio = torch.zeros(hp["n_mfcc"], 1), torch.ones(hp["n_mfcc"], 1)
        else:
            media = desvio = None
        linhas.append(dict(modelo=nome, entrada=f"{hp['n_mfcc']}x{hp['n_frames']}", **custo(model),
                           **medir_latencia(model, hp, media, desvio, repeticoes=args.repeticoes,
                                            com_librosa=args.librosa)))
    print(pd.DataFrame(linhas).round(3).to_string(index=False))
