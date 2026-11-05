"""CNN 1D — convolução ao longo do TEMPO.

Ideia: os 40 coeficientes MFCC são tratados como CANAIS e a convolução desliza só no eixo temporal
(101 frames). Cada filtro aprende um padrão espectral que se repete no tempo. Muito barata em MACs.

Arquitetura-alvo:
    4 blocos [Conv1d k=3 -> BatchNorm1d -> ReLU], 64 canais, stride 2 nos 3 últimos
    -> AdaptiveAvgPool1d(1) -> Linear(64, n_classes)

Implemente __init__ e forward, mude IMPLEMENTADO para True
e rode `pytest` para conferir o contrato de entrada/saída.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("cnn1d", familia="CNN 1D", referencia="Choi et al. 2019 (TC-ResNet) / exercício de CNNs")
class CNN1D(torch.nn.Module):
    IMPLEMENTADO = False

    def __init__(self, n_classes, n_mfcc=40, n_frames=101, canais=64):
        super().__init__()
        #TODO: definir as camadas (ver arquitetura-alvo na docstring do módulo)
        #Mantenha a última camada com o nome `self.head` (usada no few-shot do MSWC-pt)
        raise NotImplementedError("CNN1D ainda não implementado")

    def forward(self, x):
        #x: (B, n_mfcc, n_frames)  ->  logits (B, n_classes)
        raise NotImplementedError
