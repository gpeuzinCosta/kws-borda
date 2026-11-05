"""Google Speech Commands v2 (GSC v2) — o dataset núcleo do projeto (treino, validação e teste).

- 105.835 áudios de 1 s, 16 kHz, 35 palavras, ~2.600 locutores.
- Split OFICIAL por locutor (validation_list.txt / testing_list.txt): o mesmo locutor nunca aparece
  em treino e teste, então medimos generalização para vozes novas.
- Lemos os .wav com soundfile direto da pasta extraída (como na POC) em vez de
  torchaudio.datasets.SPEECHCOMMANDS — mesmo conteúdo, sem depender do backend de áudio do torchaudio.
- O MFCC de todo o dataset é calculado UMA vez e salvo em cache (.pt): treinar fica muito mais rápido
  e todos os modelos recebem exatamente a mesma entrada.

Duas TAREFAS (hp["tarefa"]):
- "gsc35": as 35 palavras. Necessária para o AudioMNIST (dígitos) e o MSWC-pt (mapeamento PT -> EN).
- "gsc12": o recorte padrão da literatura (Hello Edge, KWT) — 10 comandos + _unknown_ + _silence_.
  Torna os números comparáveis aos artigos. Montagem trazida do notebook do Luciano
  (notebooks/experimentos/Projeto_MLP.ipynb):
    * _unknown_: sorteio das outras 25 palavras, na mesma quantidade média de cada comando;
    * _silence_: trechos de 1 s dos ruídos de _background_noise_, com volume reduzido a 10%.
"""
import random
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dados.mfcc import N_AMOSTRAS, ExtratorMFCC, calcular_n_frames, ler_audio

URL_GSC = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

#As 35 palavras do GSC v2, em ordem alfabética (índice = rótulo da classe na tarefa gsc35)
CLASSES_GSC = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero",
]

#Tarefa de 12 classes (padrão Hello Edge / KWT)
COMANDOS_12 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
CLASSES_GSC12 = COMANDOS_12 + ["_unknown_", "_silence_"]

TAREFAS = {"gsc35": CLASSES_GSC, "gsc12": CLASSES_GSC12}

VOLUME_SILENCIO = 0.1     #mesmo fator do notebook do Luciano
SEMENTE_SPLIT = 42        #o sorteio de unknown/silence é fixo: todos usam os mesmos arquivos


def classes_da_tarefa(tarefa):
    if tarefa not in TAREFAS:
        raise KeyError(f"Tarefa '{tarefa}' desconhecida. Opções: {list(TAREFAS)}")
    return TAREFAS[tarefa]


def rotulo(rel, tarefa):
    """Caminho relativo -> índice da classe na tarefa. Ex.: 'cat/x.wav' -> _unknown_ na gsc12."""
    palavra = rel.split("/")[0]
    if tarefa == "gsc12" and palavra not in COMANDOS_12 and palavra != "_silence_":
        palavra = "_unknown_"
    return classes_da_tarefa(tarefa).index(palavra)


def _listar_35(raiz):
    raiz = Path(raiz)
    ler = lambda f: set((raiz / f).read_text().split())
    val, teste = ler("validation_list.txt"), ler("testing_list.txt")
    split = {"treino": [], "val": [], "teste": []}
    for c in CLASSES_GSC:
        for f in sorted(p.name for p in (raiz / c).iterdir()):
            rel = f"{c}/{f}"
            split["val" if rel in val else "teste" if rel in teste else "treino"].append(rel)
    return split


def _montar_12(raiz, split35):
    """Filtra os 10 comandos e adiciona unknown (sorteado) e silence (trechos de ruído).

    Os trechos de silêncio são representados como '_silence_/<arquivo de ruído>@<início>', e o
    `ler_trecho` sabe recortá-los. Assim tudo continua sendo uma lista de caminhos (e o cache funciona).
    """
    rng = random.Random(SEMENTE_SPLIT)
    pasta_ruido = Path(raiz) / "_background_noise_"
    ruidos = {f.name: sf.info(f).frames for f in sorted(pasta_ruido.glob("*.wav"))}

    split = {}
    for nome, arquivos in split35.items():
        comandos = [r for r in arquivos if r.split("/")[0] in COMANDOS_12]
        outros = [r for r in arquivos if r.split("/")[0] not in COMANDOS_12]
        por_classe = len(comandos) // len(COMANDOS_12)

        unknown = rng.sample(outros, min(por_classe, len(outros)))
        silencio = []
        for _ in range(por_classe):
            arq = rng.choice(list(ruidos))
            inicio = rng.randint(0, ruidos[arq] - N_AMOSTRAS)
            silencio.append(f"_silence_/{arq}@{inicio}")
        split[nome] = comandos + unknown + silencio
    return split


def listar_split(raiz, tarefa="gsc35"):
    """Separa os arquivos em treino/val/teste (listas oficiais) para a tarefa. Devolve caminhos relativos."""
    split35 = _listar_35(raiz)
    return split35 if tarefa == "gsc35" else _montar_12(raiz, split35)


def ler_trecho(raiz, rel):
    """Lê um áudio do GSC ou um trecho de silêncio ('_silence_/arquivo.wav@inicio')."""
    if rel.startswith("_silence_/"):
        arq, inicio = rel.split("/", 1)[1].split("@")
        wav, _ = sf.read(Path(raiz) / "_background_noise_" / arq, dtype="float32",
                         start=int(inicio), frames=N_AMOSTRAS)
        return VOLUME_SILENCIO * torch.from_numpy(wav)
    return ler_audio(Path(raiz) / rel)


def extrair_mfcc(raiz, arquivos, extrator, tarefa, caminho_cache=None, batch=256):
    """Lê os áudios e calcula o MFCC de todos. Se houver cache, só carrega. Devolve (X, y)."""
    if caminho_cache is not None and Path(caminho_cache).exists():
        return torch.load(caminho_cache)
    X, y = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(arquivos), batch), desc="MFCC"):
            lote = arquivos[i:i + batch]
            X.append(extrator(torch.stack([ler_trecho(raiz, rel) for rel in lote])))
            y += [rotulo(rel, tarefa) for rel in lote]
    X, y = torch.cat(X), torch.tensor(y)
    if caminho_cache is not None:
        torch.save((X, y), caminho_cache)
    return X, y


def normalizar(x, media=None, desvio=None):
    """Normaliza o MFCC (..., n_mfcc, n_frames).

    - global (media/desvio dados): estatísticas por coeficiente calculadas no TREINO. O dispositivo
      precisa guardar esses 2 x n_mfcc números.
    - por amostra (media=None): cada áudio é normalizado pela própria média/desvio ao longo do tempo
      (abordagem do notebook do Luciano). Não precisa guardar nada e se adapta ao volume do microfone.
    """
    if media is None:
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
    return (x - media) / desvio


class DatasetMFCC(Dataset):
    """Dataset em memória de MFCCs já calculados, com normalização e SpecAugment opcional."""

    def __init__(self, X, y, media, desvio, spec_augment=False):
        super().__init__()
        self.X = normalizar(X, media, desvio)
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
    max_coef = min(max_coef, n_coef // 4)     #com 10 MFCCs, no máximo 2 coeficientes
    t = torch.randint(0, max_tempo + 1, ()).item()
    t0 = torch.randint(0, n_frames - t + 1, ()).item()
    x[:, t0:t0 + t] = 0
    f = torch.randint(0, max_coef + 1, ()).item()
    f0 = torch.randint(0, n_coef - f + 1, ()).item()
    x[f0:f0 + f, :] = 0
    return x


def dados_falsos(hp, n=256):
    """MFCCs aleatórios com o formato certo — para testar o pipeline sem baixar 2,3 GB."""
    X = torch.randn(n, hp["n_mfcc"], calcular_n_frames(hp))
    y = torch.randint(0, hp["n_classes"], (n,))
    return {"treino": (X, y), "val": (X[: n // 4], y[: n // 4]), "teste": (X[: n // 4], y[: n // 4])}


def nome_cache(hp):
    """Identifica a combinação tarefa + MFCC (caches diferentes não se misturam)."""
    centro = "c" if hp.get("center", True) else "nc"
    return f"{hp['tarefa']}_{hp['n_mfcc']}x{calcular_n_frames(hp)}_{hp['janela_ms']}-{hp['passo_ms']}ms{centro}"


def carregar_gsc(hp, raiz, pasta_cache):
    """MFCC de treino/val/teste do GSC (com cache). Devolve {'treino': (X, y), 'val': ..., 'teste': ...}."""
    extrator = ExtratorMFCC.de_hparams(hp)
    split = listar_split(raiz, hp["tarefa"])
    print(hp["tarefa"], {k: len(v) for k, v in split.items()})
    return {k: extrair_mfcc(raiz, v, extrator, hp["tarefa"], Path(pasta_cache) / f"gsc_{k}_{nome_cache(hp)}.pt")
            for k, v in split.items()}


def criar_dataloaders(hp, dados):
    """Normaliza e cria os DataLoaders. Devolve (loaders, (media, desvio)).

    Com normalizacao='global', media/desvio vêm SÓ do treino (medir na validação seria vazamento).
    Com normalizacao='por_amostra', devolve (None, None).
    """
    media = desvio = None
    if hp.get("normalizacao", "global") == "global":
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
