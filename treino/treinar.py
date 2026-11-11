"""Treina um modelo no GSC v2 com o loop unificado e registra o resultado no teste.

    python treino/treinar.py --modelo mlp
    python treino/treinar.py --modelo dscnn --epocas 30 --seeds 0 1 2
    python treino/treinar.py --modelo resnet --epocas 100 --paciencia 10      # early stop (ReduceLROnPlateau)
    python treino/treinar.py --modelo mlp --dados-falsos --epocas 1     # teste rápido, sem baixar nada
    python treino/treinar.py --modelo mlp --tarefa gsc12 --mfcc hello_edge --normalizacao por_amostra
"""
import sys
from pathlib import Path

#Permite rodar direto (python treino/treinar.py) a partir da raiz do repositório
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from configs import PRESETS_MFCC, carregar_config, rotulo_experimento
from dados import carregar_gsc, criar_dataloaders, dados_falsos
from avaliacao import acuracia, custo, f1_macro, registrar
from modelos import criar_modelo
from treino import prever, treinar
from configs import escolher_device, pastas_projeto

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--learning-rate", type=float, dest="learning_rate")
    ap.add_argument("--paciencia", type=int, help="early stop: épocas sem melhora na validação (liga o ReduceLROnPlateau)")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--tarefa", choices=["gsc35", "gsc12"])
    ap.add_argument("--mfcc", choices=list(PRESETS_MFCC), help="preset de MFCC (padrão: o MFCC do próprio modelo)")
    ap.add_argument("--normalizacao", choices=["global", "por_amostra"])
    ap.add_argument("--dados-falsos", action="store_true", help="MFCC aleatório (testa o pipeline)")
    args = ap.parse_args()

    hp = carregar_config(args.modelo, epocas=args.epocas, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, seeds=args.seeds, tarefa=args.tarefa,
                         normalizacao=args.normalizacao, paciencia=args.paciencia,
                         scheduler="plateau" if args.paciencia is not None else None,
                         **PRESETS_MFCC.get(args.mfcc, {}))
    pastas = pastas_projeto()
    print(hp)

    if args.dados_falsos:
        dados, hp["num_workers"] = dados_falsos(hp), 0
    else:
        dados = carregar_gsc(hp, pastas["dados"] / "gsc", pastas["cache"])
    loaders, _ = criar_dataloaders(hp, dados)
    device = escolher_device()

    for seed in hp["seeds"]:
        print(f"\n===== {hp['modelo']} | seed {seed} | device {device} =====")
        model, hist = treinar(criar_modelo(hp["modelo"], hp), loaders, hp, seed=seed,
                              pasta_runs=pastas["runs"], device=device)
        y_pred, y_true = prever(model, loaders["teste"])
        registrar(pastas["resultados"] / "resultados.csv", hp["modelo"], seed,
                  dataset="falso" if args.dados_falsos else "gsc_v2", experimento=rotulo_experimento(hp),
                  acc=acuracia(y_pred, y_true), f1=f1_macro(y_pred, y_true),
                  epocas=hist["epocas_treinadas"], **custo(model.cpu()), run=hist.get("pasta"))
