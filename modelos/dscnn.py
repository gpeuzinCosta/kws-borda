"""DS-CNN-S baseada no artigo Hello Edge.

Arquitetura:

    entrada (B, 1, tempo, frequência)

    -> Conv2d comum:
       1 -> 64 canais
       kernel 10x4
       stride 2x2
       BatchNorm
       ReLU

    -> 4 blocos:
       DepthwiseConv 3x3
       BatchNorm
       ReLU
       PointwiseConv 1x1
       BatchNorm
       ReLU

    -> média global
    -> Linear
"""

import torch

from modelos import registrar_modelo


class BlocoDepthwiseSeparable(torch.nn.Module):

    def __init__(self, canais):
        super().__init__()

        # Um kernel 3x3 separado para cada canal
        self.depthwise = torch.nn.Conv2d(
            in_channels=canais,
            out_channels=canais,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=canais,
            bias=False
        )

        self.bn_depthwise = torch.nn.BatchNorm2d(
            canais
        )

        # Convolução comum 1x1 para misturar os canais
        self.pointwise = torch.nn.Conv2d(
            in_channels=canais,
            out_channels=canais,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.bn_pointwise = torch.nn.BatchNorm2d(
            canais
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = self.relu(x)

        return x


@registrar_modelo(
    "dscnn",
    familia="DS-CNN",
    referencia=(
        "Zhang et al. 2017 — "
        "Hello Edge (arXiv:1711.07128)"
    )
)
class DSCNN(torch.nn.Module):

    IMPLEMENTADO = True

    def __init__(
        self,
        n_classes,
        n_mfcc=40,
        n_frames=101,
        canais=64,
        n_blocos=4
    ):
        super().__init__()

        def calcular_padding_same(
            tamanho,
            kernel,
            stride
        ):
            tamanho_saida = (
                tamanho + stride - 1
            ) // stride

            padding_total = max(
                (
                    tamanho_saida - 1
                ) * stride
                + kernel
                - tamanho,
                0
            )

            padding_inicio = (
                padding_total // 2
            )

            padding_fim = (
                padding_total
                - padding_inicio
            )

            return (
                padding_inicio,
                padding_fim
            )

        # O artigo usa padding SAME.
        # PyTorch exige padding manual para kernel par
        # combinado com stride maior que 1.

        padding_tempo = calcular_padding_same(
            tamanho=n_frames,
            kernel=10,
            stride=2
        )

        padding_frequencia = calcular_padding_same(
            tamanho=n_mfcc,
            kernel=4,
            stride=2
        )

        # ZeroPad2d:
        # esquerda, direita, cima, baixo
        self.padding_inicial = torch.nn.ZeroPad2d(
            (
                padding_frequencia[0],
                padding_frequencia[1],
                padding_tempo[0],
                padding_tempo[1]
            )
        )

        # Primeira convolução comum
        self.conv_inicial = torch.nn.Conv2d(
            in_channels=1,
            out_channels=canais,
            kernel_size=(10, 4),
            stride=(2, 2),
            bias=False
        )

        self.bn_inicial = torch.nn.BatchNorm2d(
            canais
        )

        self.relu = torch.nn.ReLU()

        self.blocos = torch.nn.Sequential(
            *[
                BlocoDepthwiseSeparable(
                    canais=canais
                )
                for _ in range(n_blocos)
            ]
        )

        self.media_global = (
            torch.nn.AdaptiveAvgPool2d(
                output_size=(1, 1)
            )
        )

        # O nome head é usado pelo few-shot do projeto
        self.head = torch.nn.Linear(
            in_features=canais,
            out_features=n_classes
        )

    def forward(self, x):

        # Entrada do pipeline:
        # (B, n_mfcc, n_frames)

        # Artigo:
        # (B, 1, tempo, frequência)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)

        x = self.padding_inicial(x)

        x = self.conv_inicial(x)
        x = self.bn_inicial(x)
        x = self.relu(x)

        x = self.blocos(x)

        x = self.media_global(x)

        # (B, canais, 1, 1)
        # -> (B, canais)
        x = x.flatten(1)

        return self.head(x)
