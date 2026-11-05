"""Baixa e extrai os datasets do projeto.

    python dados/baixar.py --dataset gsc --destino data/          # 2,3 GB
    python dados/baixar.py --dataset demand --destino data/       # ruído, 8 ambientes
    python dados/baixar.py --dataset audiomnist --destino data/
    python dados/baixar.py --dataset mswc --destino data/         # 1,1 GB (só as palavras usadas)

No Colab: --arquivos <pasta no Drive> guarda o .tar.gz/.zip no Drive (baixa UMA vez) e extrai no disco local.
"""
import sys
from pathlib import Path

#Permite rodar direto (python dados/baixar.py) a partir da raiz do repositório
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from dados.audiomnist import URL_AUDIOMNIST
from dados.mswc_pt import PT2EN, URL_MSWC_PT
from dados.ruido import AMBIENTES_DEMAND, URL_DEMAND
from dados.speech_commands import URL_GSC


def baixar(url, arquivo):
    arquivo = Path(arquivo)
    if arquivo.exists():
        print(f"já existe: {arquivo}")
        return arquivo
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    print(f"baixando {url} -> {arquivo}")
    tmp = arquivo.with_suffix(arquivo.suffix + ".parcial")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(arquivo)     #só vira o nome final quando o download terminou
    return arquivo


def gsc(destino, arquivos):
    pasta = destino / "gsc"
    if (pasta / "testing_list.txt").exists():
        print(f"GSC já extraído em {pasta}")
        return
    tar = baixar(URL_GSC, arquivos / "speech_commands_v0.02.tar.gz")
    pasta.mkdir(parents=True, exist_ok=True)
    print("extraindo GSC (1-2 min)...")
    with tarfile.open(tar) as t:
        t.extractall(pasta)


def demand(destino, arquivos):
    pasta = destino / "demand"
    for amb in AMBIENTES_DEMAND:
        if (pasta / amb / "ch01.wav").exists():
            continue
        z = baixar(URL_DEMAND.format(ambiente=amb), arquivos / f"demand_{amb}_16k.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(pasta)


def audiomnist(destino, arquivos):
    pasta = destino / "audiomnist"
    if not pasta.exists():
        subprocess.run(["git", "clone", "--depth", "1", URL_AUDIOMNIST, str(pasta)], check=True)


def mswc(destino, arquivos):
    pasta = destino / "mswc"
    tar = baixar(URL_MSWC_PT, arquivos / "mswc_pt.tar.gz")
    #Extrai só as palavras de interesse (o corpus inteiro tem 5.664 palavras)
    with tarfile.open(tar) as t:
        membros = [m for m in t.getmembers()
                   if any(m.name.startswith(f"pt/clips/{w}/") for w in PT2EN)]
        t.extractall(pasta, members=membros)


DATASETS = {"gsc": gsc, "demand": demand, "audiomnist": audiomnist, "mswc": mswc}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS) + ["todos"], default="gsc")
    ap.add_argument("--destino", default="data", help="onde extrair (no Colab: /content/data)")
    ap.add_argument("--arquivos", default=None, help="onde guardar os .tar.gz/.zip (no Colab: pasta no Drive)")
    args = ap.parse_args()
    destino = Path(args.destino)
    arquivos = Path(args.arquivos) if args.arquivos else destino / "_downloads"
    for nome in (DATASETS if args.dataset == "todos" else [args.dataset]):
        DATASETS[nome](destino, arquivos)
