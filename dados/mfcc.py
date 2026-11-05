"""Pré-processamento único do áudio para TODOS os modelos.

Pipeline:
    áudio qualquer -> mono 16 kHz -> exatamente 1 s (16000 amostras) -> MFCC 40 coef
    janela de 30 ms (480 amostras) e passo de 10 ms (160 amostras) -> 101 frames
    saída: tensor (B, 40, 101)

Por que MFCC? A forma de onda crua tem 16000 números por segundo e muita informação irrelevante para
reconhecer a palavra (fase, ruído de alta frequência). O MFCC resume o espectro em poucas dezenas de
coeficientes por janela, na escala Mel (parecida com a audição humana), e reduz muito a entrada da rede.
O custo de calcular o MFCC entra na hipótese (b) do projeto, por isso fica isolado aqui.
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000          #taxa de amostragem alvo
N_AMOSTRAS = SR     #1 segundo de áudio


class ExtratorMFCC(torch.nn.Module):
    """Transforma waveform (B, N) ou (1, N) em MFCC (B, n_mfcc, frames).

    É um nn.Module para poder ser medido (latência) e movido de device igual a uma rede.
    """

    def __init__(self, n_mfcc=40, janela_ms=30, passo_ms=10, n_mels=40, f_min=20, f_max=7600, sample_rate=SR,
                 center=True):
        super().__init__()
        n_fft = int(sample_rate * janela_ms / 1000)    #30 ms -> 480 amostras
        hop = int(sample_rate * passo_ms / 1000)       #10 ms -> 160 amostras
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            melkwargs=dict(n_fft=n_fft, win_length=n_fft, hop_length=hop,
                           n_mels=n_mels, f_min=f_min, f_max=f_max, center=center),
        )
        self.config = dict(n_mfcc=n_mfcc, n_fft=n_fft, hop=hop, n_mels=n_mels, f_min=f_min, f_max=f_max,
                           center=center)

    @classmethod
    def de_hparams(cls, hp):
        """Cria o extrator a partir do dicionário de hiperparâmetros."""
        return cls(n_mfcc=hp["n_mfcc"], janela_ms=hp["janela_ms"], passo_ms=hp["passo_ms"],
                   n_mels=hp["n_mels"], f_min=hp["f_min"], f_max=hp["f_max"], sample_rate=hp["sample_rate"],
                   center=hp.get("center", True))

    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return self.mfcc(wav)


def calcular_n_frames(hp):
    """Nº de janelas do MFCC para 1 s de áudio.

    center=True (padrão do torchaudio): completa as bordas -> 1 + N // passo         (16000/160 -> 101)
    center=False (Hello Edge):          só janelas inteiras -> 1 + (N - janela) // passo  (640/320 -> 49)
    """
    n = int(hp["sample_rate"] * hp["duracao_s"])
    n_fft = int(hp["sample_rate"] * hp["janela_ms"] / 1000)
    hop = int(hp["sample_rate"] * hp["passo_ms"] / 1000)
    return 1 + n // hop if hp.get("center", True) else 1 + (n - n_fft) // hop


def fix_length(wav, n=N_AMOSTRAS):
    """(1, T) -> (1, n): completa com zeros se curto, corta no centro se longo."""
    T = wav.shape[-1]
    if T < n:
        return torch.nn.functional.pad(wav, (0, n - T))
    inicio = (T - n) // 2
    return wav[..., inicio:inicio + n]


def to_16k_mono(wav, sr):
    """Garante 1 canal e 16 kHz (AudioMNIST e MSWC vêm em 48 kHz)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def ler_audio(caminho):
    """Lê wav/opus com soundfile (evita depender do backend de I/O do torchaudio) -> tensor (N_AMOSTRAS,)."""
    wav, sr = sf.read(caminho, dtype="float32", always_2d=True)    #(T, canais)
    wav = torch.from_numpy(np.ascontiguousarray(wav.T))           #(canais, T)
    return fix_length(to_16k_mono(wav, sr))[0]


def mfcc_librosa(wav_np, n_mfcc=40, n_fft=480, hop=160, n_mels=40, f_min=20, f_max=7600):
    """Mesma configuração via librosa — para comparar o custo de IMPLEMENTAÇÃO do MFCC (hipótese b).
    """
    import librosa
    return librosa.feature.mfcc(y=wav_np, sr=SR, n_mfcc=n_mfcc, n_fft=n_fft, win_length=n_fft,
                                hop_length=hop, n_mels=n_mels, fmin=f_min, fmax=f_max)
