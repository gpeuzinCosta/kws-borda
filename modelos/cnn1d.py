"""CNN 1D — convolução ao longo do TEMPO.

Ideia: os coeficientes MFCC são tratados como CANAIS e a convolução desliza só no eixo temporal.
Cada filtro aprende um padrão espectral que se repete no tempo. Muito barata em MACs.

Arquitetura:
    4 blocos [Conv1d k=3 -> BatchNorm1d -> ReLU], stride 2 nos 3 últimos
    -> AdaptiveAvgPool1d(1) -> Linear(canais, n_classes)

Configuração vencedora da varredura (notebooks/experimentos/playground_cnn1d_guilherme.ipynb, 3 seeds):
entrada hello_edge (10 x 49) + 48 canais = o MENOR modelo a menos de 2 p.p. do baseline (40 x 101, 64 canais).
24.275 parâmetros (23,7 KB em int8), 383.280 MACs, acurácia de validação 92,46% (baseline: 93,51%).
Os hiperparâmetros estão em configs/cnn1d.yaml.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo("cnn1d", familia="CNN 1D", referencia="Choi et al. 2019 (TC-ResNet) / exercício de CNNs")
class CNN1D(torch.nn.Module):
    """
    CNN 1D: convolução ao longo do TEMPO.
    Os n_mfcc coeficientes são tratados como CANAIS e o filtro desliza só no eixo temporal (as 101 janelas).
    Cada filtro aprende um padrão espectral que se repete no tempo.
    """
    IMPLEMENTADO = True

    #n_frames não é usado: o pooling global aceita qualquer comprimento.
    def __init__(self, n_classes, n_mfcc=40, n_frames=101, canais=64, kernel_size=3):
        super().__init__()

        k = kernel_size
        p = k // 2   #Padding para o tempo não encolher na borda

        #x chega como (B, n_mfcc, n_frames): os 40 coeficientes já são os canais que o nn.Conv1d espera,
        #então não precisa de permute

        #Bloco 1: 101 -> 101 (stride 1, mantém o tempo)
        #40,64
        self.conv1 = torch.nn.Conv1d(n_mfcc, canais, k, stride=1, padding=p, bias=False) #o BatchNorm anula qualquer bias
        self.bn1   = torch.nn.BatchNorm1d(canais)

        #Blocos 2, 3 e 4: stride 2, o tempo cai pela metade a cada bloco
        #64,64
        self.conv2 = torch.nn.Conv1d(canais, canais, k, stride=2, padding=p, bias=False)   # 101 -> 51
        self.bn2   = torch.nn.BatchNorm1d(canais)
        self.conv3 = torch.nn.Conv1d(canais, canais, k, stride=2, padding=p, bias=False)   # 51 -> 26
        self.bn3   = torch.nn.BatchNorm1d(canais)
        self.conv4 = torch.nn.Conv1d(canais, canais, k, stride=2, padding=p, bias=False)   # 26 -> 13
        self.bn4   = torch.nn.BatchNorm1d(canais)

        self.relu = torch.nn.ReLU()

        #Média das janelas finais
        self.gap = torch.nn.AdaptiveAvgPool1d(1)

        self.head = torch.nn.Linear(canais, n_classes)

    def forward(self, input_tensor):
        x = input_tensor                              # (B, 40, 101)

        x = self.relu(self.bn1(self.conv1(x)))        # (B, 64, 101)
        x = self.relu(self.bn2(self.conv2(x)))        # (B, 64, 51)
        x = self.relu(self.bn3(self.conv3(x)))        # (B, 64, 26)
        x = self.relu(self.bn4(self.conv4(x)))        # (B, 64, 13)

        x = self.gap(x)                               # (B, 64, 1)
        x = x.view(x.size(0), -1)                     # (B, 64)

        return self.head(x)                           # (B, 35)
