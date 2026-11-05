"""ResNet pequena — conexões residuais (estilo res8).

Ideia: trata o MFCC como uma "imagem" 1 x 40 x 101 (Conv2d). Os atalhos residuais (x + F(x)) deixam
o gradiente fluir e permitem redes mais profundas com poucos canais.

Arquitetura-alvo (POC: 112 mil parâmetros, 37 M MACs, 94,5%):
    Conv2d(1, 45, 3) -> BN -> ReLU -> AvgPool(3, 4)
    -> 3 blocos residuais [Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN] + atalho
    -> média global -> Linear(45, n_classes)
Referência pronta: classe ResNet em kws-borda-poc/models.py

STATUS: a fazer — esqueleto. Implemente __init__ e forward, mude IMPLEMENTADO para True
e rode `pytest` para conferir o contrato de entrada/saída.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("resnet", familia="ResNet", referencia="He et al. 2016; Tang & Lin 2018 (res8)")
class ResNet(torch.nn.Module):
    IMPLEMENTADO = False

    def __init__(self, n_classes, n_mfcc=40, n_frames=101, canais=45, n_blocos=3):
        super().__init__()
        #TODO: definir as camadas (ver arquitetura-alvo na docstring do módulo)
        #Mantenha a última camada com o nome `self.head` (usada no few-shot do MSWC-pt)
        raise NotImplementedError("ResNet ainda não implementado")

    def forward(self, x):
        #x: (B, n_mfcc, n_frames)  ->  logits (B, n_classes)
        raise NotImplementedError
