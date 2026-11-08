"""ResNet pequena — conexões residuais (Tang & Lin 2018: res8, res15 e res26).

Ideia: trata o MFCC como uma "imagem" 1 x n_mfcc x n_frames (Conv2d 3x3). Os atalhos residuais (x + F(x)) deixam
o gradiente fluir e permitem redes mais profundas com poucos canais.

Arquitetura:
    Conv2d(1, canais, 3) -> BN -> ReLU -> AvgPool(pool) (ou nada, se pool=None)
    -> n_blocos blocos residuais [Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN] + atalho -> ReLU
    -> (conv_extra: mais uma Conv 3x3 -> BN -> ReLU)
    -> média global -> Linear(canais, n_classes)

As famílias do artigo:
    res8:  canais 45, n_blocos 3,  pool (3, 4)
    res15: canais 45, n_blocos 6,  sem pool, convoluções dilatadas (1, 1, 2, 2, 2, 4, ...) e conv extra
    res26: canais 45, n_blocos 12, pool (2, 2)
    as versões "-narrow" usam 19 canais.

Escrita e testada no playground da ResNet (notebooks/experimentos/playground_resnet_guilherme.ipynb). A vencedora
da varredura (res15-narrow + MFCC hello_edge) está em configs.HP_MODELOS['resnet'].
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("resnet", familia="ResNet", referencia="He et al. 2016; Tang & Lin 2018 (res8)")
class ResNet(torch.nn.Module):
    IMPLEMENTADO = True

    def __init__(self, n_classes, n_mfcc=40, n_frames=101, canais=45, n_blocos=3,
                 pool=(3, 4), dilatacao=False, conv_extra=False):      #pool, dilatacao, conv_extra
        super().__init__()

        k = 3   #kernel 3x3 em todas as camadas, como no Tang & Lin

        #Entrada: 1 canal -> canais
        self.conv0 = torch.nn.Conv2d(1, canais, k, padding=1, bias=False) #o Batch norm vai anular qualquer bias, pois subtrai a media do batch
        self.bn0   = torch.nn.BatchNorm2d(canais)

        #NOVO: res8 usa pool 3x4, res26 usa 2x2 e res15 não usa pooling
        if pool is None:
            self.pool = torch.nn.Identity()
        else:
            self.pool = torch.nn.AvgPool2d(pool)

        #Blocos residuais: 2 convs por bloco (+1 conv extra no res15)
        n_convs = 2 * n_blocos
        if conv_extra:
            n_convs = n_convs + 1
        self.convs = torch.nn.ModuleList()
        self.bns   = torch.nn.ModuleList()
        for i in range(1, n_convs + 1):
            if dilatacao:
                d = 2 ** (i // 3)                    #res15: 1,1,2,2,2,4,4,4,8,8,8,16,16
            else:
                d = 1
            #padding = d mantém o tamanho do mapa, para a soma com o atalho bater
            self.convs.append(torch.nn.Conv2d(canais, canais, k, padding=d, dilation=d, bias=False))
            self.bns.append(torch.nn.BatchNorm2d(canais))

        self.relu = torch.nn.ReLU()
        self.gap  = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(canais, n_classes)    #última camada sempre com o nome `head`

    def forward(self, x):
        #x: (B, n_mfcc, n_frames) -> logits (B, n_classes)
        x = x.unsqueeze(1)      #(B, 1, n_mfcc, n_frames): 1 canal, como uma imagem em tons de cinza

        x = self.relu(self.bn0(self.conv0(x))) #primeira conv com norm e relu
        x = self.pool(x)

        #cada bloco = conv -> norm -> ReLu -> conv -> norm, soma o atalho, ReLu
        for b in range(len(self.convs) // 2):
            atalho = x
            y = self.relu(self.bns[2 * b](self.convs[2 * b](x)))
            y = self.bns[2 * b + 1](self.convs[2 * b + 1](y))
            x = self.relu(y + atalho)

        #conv extra do res15 (sobra 1 conv quando o total é ímpar)
        if len(self.convs) % 2 == 1:
            x = self.relu(self.bns[-1](self.convs[-1](x)))

        #Descarta posicoes e mantem so uma intensidade media para a fully connected ter menos parametros
        x = self.gap(x)                               # (B, canais, 1, 1)
        x = x.view(x.size(0), -1)                     # (B, canais)

        return self.head(x)                           # (B, n_classes) — logits crus
