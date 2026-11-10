"""Atenção — Keyword Transformer (KWT-tiny).

Ideia: cada frame do MFCC (vetor de 40 coeficientes) vira um TOKEN, como uma palavra numa frase ou um
patch no Vision Transformer. A autoatenção deixa cada frame olhar para todos os outros, capturando
contexto global em uma camada só. Um token [CLS] aprendido resume o áudio para a classificação.

Arquitetura-alvo:
    Linear(40 -> d=64) por frame + token CLS + embedding posicional (102 posições)
    -> 2 camadas TransformerEncoder (2 cabeças, FFN 128, pre-norm) -> LayerNorm(CLS) -> Linear(64, n_classes)
Atenção ao medir MACs: em eval() o PyTorch usa um kernel fundido invisível ao contador (ver evaluation/custo.py).
Referência: notebook "Vision Transformer do zero" da disciplina

Implemente __init__ e forward, mude IMPLEMENTADO para True
e rode `pytest` para conferir o contrato de entrada/saída.
"""
import torch

from modelos import registrar_modelo


@registrar_modelo(
    "atencao",
    familia="Atenção",
    referencia="Berg et al. 2021 — KWT (arXiv:2104.00769)"
)
class Atencao(torch.nn.Module):
    IMPLEMENTADO = True

    def __init__(
        self,
        n_classes,
        n_mfcc=40,
        n_frames=101,
        d_model=64,
        n_camadas=3,
        n_heads=1,
        dropout=0.0
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model deve ser divisível por n_heads"
            )

        # Cada janela do MFCC vira um token de dimensão d_model.
        self.projecao = torch.nn.Linear(
            n_mfcc,
            d_model
        )

        # Um único parâmetro CLS, compartilhado entre todos os áudios.
        self.cls_token = torch.nn.Parameter(
            torch.empty(1, 1, d_model)
        )

        # Uma posição para o CLS e uma para cada janela temporal.
        self.pos_embedding = torch.nn.Parameter(
            torch.empty(
                1,
                n_frames + 1,
                d_model
            )
        )

        camada_encoder = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,  # 64 -> 256 -> 64
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False  # PostNorm, como no KWT
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=camada_encoder,
            num_layers=n_camadas
        )

        torch.nn.init.trunc_normal_(
            self.cls_token,
            std=0.02
        )

        torch.nn.init.trunc_normal_(
            self.pos_embedding,
            std=0.02
        )

        # Deve se chamar head por causa do few-shot do projeto.
        self.head = torch.nn.Linear(
            d_model,
            n_classes
        )

    def forward(self, x):
        # x: (B, n_mfcc, n_frames)

        batch_size = x.size(0)

        # Cada janela temporal deve ser um token.
        # (B, n_mfcc, n_frames)
        # ->
        # (B, n_frames, n_mfcc)
        x = x.transpose(1, 2)

        # Projeta cada janela:
        # (B, n_frames, n_mfcc)
        # ->
        # (B, n_frames, d_model)
        x = self.projecao(x)

        # Copia o mesmo parâmetro CLS para cada áudio do batch.
        # (1, 1, d_model)
        # ->
        # (B, 1, d_model)
        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        # (B, n_frames + 1, d_model)
        x = torch.cat(
            [cls, x],
            dim=1
        )

        if x.size(1) > self.pos_embedding.size(1):
            raise ValueError(
                f"A entrada possui {x.size(1) - 1} frames, "
                f"mas o modelo foi criado para no máximo "
                f"{self.pos_embedding.size(1) - 1}."
            )

        # Mesmo embedding posicional para a mesma posição
        # em todos os áudios.
        x = x + self.pos_embedding[:, :x.size(1)]

        # Doze blocos encoder em série.
        # O formato continua (B, n_frames + 1, d_model).
        x = self.encoder(x)

        # Um CLS final diferente para cada áudio.
        # (B, n_frames + 1, d_model)
        # ->
        # (B, d_model)
        cls_final = x[:, 0, :]

        # (B, d_model) -> (B, n_classes)
        logits = self.head(cls_final)

        return logits
