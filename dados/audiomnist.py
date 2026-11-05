"""AudioMNIST — dígitos falados por OUTROS falantes, em outras salas.

Achado da POC: 30.000 áudios, 60 falantes, 48 kHz. Mesmo microfone para todos; o que varia é a SALA
e o SOTAQUE (57 de 60 não nativos de inglês). Então o teste é "generaliza para outros falantes/sotaques/salas",
não "microfones diferentes". Os 10 dígitos existem no GSC (zero..nine), então o modelo treinado no GSC
pode ser avaliado direto, sem re-treino.
"""
from pathlib import Path
import random

URL_AUDIOMNIST = "https://github.com/soerenab/AudioMNIST"
DIGITOS_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def listar_audiomnist(raiz, por_falante=None, seed=0):
    """Lista (arquivos, rótulos 0-9). O dígito é o 1º caractere do nome: <digito>_<falante>_<rep>.wav

    por_falante: limita o nº de gravações por falante (a POC usou 100 para caber no tempo).
    """
    rng = random.Random(seed)
    por_spk = {}
    for f in sorted(Path(raiz, "data").glob("*/*.wav")):
        por_spk.setdefault(f.parent.name, []).append(f)
    arquivos = []
    for fs in por_spk.values():
        arquivos += fs if por_falante is None else rng.sample(fs, min(por_falante, len(fs)))
    rotulos = [int(f.name[0]) for f in arquivos]
    return arquivos, rotulos
