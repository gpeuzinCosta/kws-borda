"""MLP — o baseline mais simples (exemplo de referência de como implementar um modelo no projeto).

Achata o MFCC (40 x 101 = 4.040 números) e passa por camadas densas.
Não tem nenhuma noção de tempo nem de vizinhança: cada coeficiente em cada frame é um peso separado.
Por isso tende a ter MUITOS parâmetros e acurácia mais baixa que as redes convolucionais.
É o contraexemplo útil para a hipótese (a): tamanho não é tudo, a arquitetura importa.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("mlp", familia="MLP", referencia="Bengio et al. 2003 (exercício da disciplina)")
class MLP(torch.nn.Module):

    def __init__(self, n_classes, n_mfcc=40, n_frames=101, hidden=64, n_camadas=2, dropout=0.1, batchnorm=True):
        super().__init__()
        camadas = [torch.nn.Flatten()]       #(B, 40, 101) -> (B, 4040)
        entrada = n_mfcc * n_frames
        for _ in range(n_camadas):
            #BatchNorm estabiliza o treino; Dropout regulariza (a MLP decora fácil)
            camadas.append(torch.nn.Linear(entrada, hidden))
            if batchnorm:
                camadas.append(torch.nn.BatchNorm1d(hidden))
            camadas += [torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            entrada = hidden
        self.corpo = torch.nn.Sequential(*camadas)
        self.head = torch.nn.Linear(hidden, n_classes)   #'head' = última camada (trocada no few-shot do MSWC)

    def forward(self, x):
        return self.head(self.corpo(x))
