"""Atenção — Keyword Transformer (KWT-tiny).

Ideia: cada frame do MFCC (vetor de 40 coeficientes) vira um TOKEN, como uma palavra numa frase ou um
patch no Vision Transformer. A autoatenção deixa cada frame olhar para todos os outros, capturando
contexto global em uma camada só. Um token [CLS] aprendido resume o áudio para a classificação.

Arquitetura-alvo:
    Linear(40 -> d=64) por frame + token CLS + embedding posicional (102 posições)
    -> 2 camadas TransformerEncoder (2 cabeças, FFN 128, pre-norm) -> LayerNorm(CLS) -> Linear(64, n_classes)
Atenção ao medir MACs: em eval() o PyTorch usa um kernel fundido invisível ao contador (ver evaluation/custo.py).
Referência: notebook "Vision Transformer do zero" da disciplina

STATUS: a fazer — esqueleto. Implemente __init__ e forward, mude IMPLEMENTADO para True
e rode `pytest` para conferir o contrato de entrada/saída.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("atencao", familia="Atenção", referencia="Vaswani et al. 2017; Berg et al. 2021 — KWT (arXiv:2104.00769)")
class Atencao(torch.nn.Module):
    IMPLEMENTADO = False

    def __init__(self, n_classes, n_mfcc=40, n_frames=101, d_model=64, n_camadas=2, n_heads=2, dropout=0.1):
        super().__init__()
        #TODO: definir as camadas (ver arquitetura-alvo na docstring do módulo)
        #Mantenha a última camada com o nome `self.head` (usada no few-shot do MSWC-pt)
        raise NotImplementedError("Atencao ainda não implementado")

    def forward(self, x):
        #x: (B, n_mfcc, n_frames)  ->  logits (B, n_classes)
        raise NotImplementedError
