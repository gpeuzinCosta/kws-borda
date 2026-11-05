"""DS-CNN — convoluções separáveis em profundidade (Hello Edge, DS-CNN-S).

Ideia: uma Conv 3x3 comum com C canais custa C*C*9 pesos. A separável divide em
  (1) depthwise: um filtro 3x3 POR canal (C*9)  +  (2) pointwise: Conv 1x1 que mistura canais (C*C).
~8-9x menos parâmetros e MACs com acurácia parecida — foi feita para microcontroladores.

Arquitetura-alvo (POC: 25 mil parâmetros, 22 M MACs, 93,1%):
    entrada (B, 1, T, F) -> Conv2d(1, 64, (10, 4), stride 2) -> BN -> ReLU
    -> 4 x [DepthwiseConv 3x3 -> BN -> ReLU -> PointwiseConv 1x1 -> BN -> ReLU]
    -> média global -> Linear(64, n_classes)
Referência pronta: classe DSCNN em kws-borda-poc/models.py

STATUS: 🔜 a fazer — esqueleto. Implemente __init__ e forward, mude IMPLEMENTADO para True
e rode `pytest` para conferir o contrato de entrada/saída.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("dscnn", familia="DS-CNN", referencia="Zhang et al. 2017 — Hello Edge (arXiv:1711.07128)")
class DSCNN(torch.nn.Module):
    IMPLEMENTADO = False

    def __init__(self, n_classes, n_mfcc=40, n_frames=101, canais=64, n_blocos=4):
        super().__init__()
        #TODO: definir as camadas (ver arquitetura-alvo na docstring do módulo)
        #Mantenha a última camada com o nome `self.head` (usada no few-shot do MSWC-pt)
        raise NotImplementedError("DSCNN ainda não implementado")

    def forward(self, x):
        #x: (B, n_mfcc, n_frames)  ->  logits (B, n_classes)
        raise NotImplementedError
