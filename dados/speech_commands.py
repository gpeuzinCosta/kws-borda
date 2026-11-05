"""Google Speech Commands v2 (GSC v2) — o dataset núcleo do projeto (treino, validação e teste).

- 105.835 áudios de 1 s, 16 kHz, 35 palavras, ~2.600 locutores.
- Split OFICIAL por locutor (validation_list.txt / testing_list.txt): o mesmo locutor nunca aparece
  em treino e teste, então medimos generalização para vozes novas.
- Lemos os .wav com soundfile direto da pasta extraída (como na POC) em vez de
  torchaudio.datasets.SPEECHCOMMANDS — mesmo conteúdo, sem depender do backend de áudio do torchaudio.
- O MFCC de todo o dataset é calculado UMA vez e salvo em cache (.pt): treinar fica muito mais rápido
  e todos os modelos recebem exatamente a mesma entrada.
"""
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dados.mfcc import ExtratorMFCC, ler_audio

URL_GSC = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

#As 35 palavras do GSC v2, em ordem alfabética (índice = rótulo da classe)
CLASSES_GSC = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero",
]


def listar_split(raiz):
    """Separa os arquivos em treino/val/teste usando as listas oficiais. Devolve caminhos relativos."""
    raiz = Path(raiz)
    ler = lambda f: set((raiz / f).read_text().split())
    val, teste = ler("validation_list.txt"), ler("testing_list.txt")
    split = {"treino": [], "val": [], "teste": []}
    for c in CLASSES_GSC:
        for f in sorted(os.listdir(raiz / c)):
            rel = f"{c}/{f}"
            split["val" if rel in val else "teste" if rel in teste else "treino"].append(rel)
    return split


def extrair_mfcc(raiz, arquivos, extrator, caminho_cache=None, batch=256):
    """Lê os áudios e calcula o MFCC de todos. Se houver cache, só carrega. Devolve (X, y)."""
    if caminho_cache is not None and Path(caminho_cache).exists():
        return torch.load(caminho_cache)
    X, y = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(arquivos), batch), desc="MFCC"):
            lote = arquivos[i:i + batch]
            wavs = torch.stack([ler_audio(Path(raiz) / rel) for rel in lote])
            X.append(extrator(wavs))
            y += [CLASSES_GSC.index(rel.split("/")[0]) for rel in lote]
    X, y = torch.cat(X), torch.tensor(y)
    if caminho_cache is not None:
        torch.save((X, y), caminho_cache)
    return X, y


class DatasetMFCC(Dataset):
    """Dataset em memória de MFCCs já calculados, com normalização e SpecAugment opcional.

    A normalização usa média/desvio do TREINO por coeficiente — medir na validação seria vazamento.
    """

    def __init__(self, X, y, media, desvio, spec_augment=False):
        super().__init__()
        self.X = (X - media) / desvio
        self.y = y
        self.spec_augment = spec_augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.spec_augment:
            x = spec_augment(x)
        return x, self.y[idx]


def spec_augment(x, max_tempo=10, max_coef=7):
    """SpecAugment leve: zera uma faixa de frames e uma faixa de coeficientes (igual para todos os modelos).

    Força a rede a não depender de um único trecho do áudio — regularização barata.
    """
    x = x.clone()
    n_coef, n_frames = x.shape
    t = torch.randint(0, max_tempo + 1, ()).item()
    t0 = torch.randint(0, n_frames - t + 1, ()).item()
    x[:, t0:t0 + t] = 0
    f = torch.randint(0, max_coef + 1, ()).item()
    f0 = torch.randint(0, n_coef - f + 1, ()).item()
    x[f0:f0 + f, :] = 0
    return x


def dados_falsos(hp, n=256):
    """MFCCs aleatórios com o formato certo — para testar o pipeline sem baixar 2,3 GB."""
    n_frames = int(hp["duracao_s"] * 1000 / hp["passo_ms"]) + 1
    X = torch.randn(n, hp["n_mfcc"], n_frames)
    y = torch.randint(0, hp["n_classes"], (n,))
    return {"treino": (X, y), "val": (X[: n // 4], y[: n // 4]), "teste": (X[: n // 4], y[: n // 4])}


def carregar_gsc(hp, raiz, pasta_cache):
    """MFCC de treino/val/teste do GSC (com cache). Devolve {'treino': (X, y), 'val': ..., 'teste': ...}."""
    extrator = ExtratorMFCC.de_hparams(hp)
    split = listar_split(raiz)
    print({k: len(v) for k, v in split.items()})
    sufixo = f"{hp['n_mfcc']}x{hp['passo_ms']}ms"
    return {k: extrair_mfcc(raiz, v, extrator, Path(pasta_cache) / f"gsc_{k}_{sufixo}.pt")
            for k, v in split.items()}


def criar_dataloaders(hp, dados):
    """Normaliza com estatísticas do treino e cria os DataLoaders. Devolve (loaders, (media, desvio))."""
    Xtr = dados["treino"][0]
    media = Xtr.mean(dim=(0, 2), keepdim=True)[0]      #(n_mfcc, 1): uma média por coeficiente
    desvio = Xtr.std(dim=(0, 2), keepdim=True)[0]
    loaders = {}
    for nome, (X, y) in dados.items():
        eh_treino = nome == "treino"
        ds = DatasetMFCC(X, y, media, desvio, spec_augment=eh_treino and hp.get("spec_augment", False))
        loaders[nome] = DataLoader(ds, batch_size=hp["batch_size"], shuffle=eh_treino,
                                   num_workers=hp.get("num_workers", 0),
                                   pin_memory=torch.cuda.is_available())
    return loaders, (media, desvio)
