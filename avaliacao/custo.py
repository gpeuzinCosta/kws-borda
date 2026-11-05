"""Custo computacional de um modelo — quanto ele "pesa" para um dispositivo restrito.

- Parâmetros: nº de pesos treináveis. Tamanho em memória = params x 4 bytes (fp32) ou x 1 byte (int8 quantizado).
- MACs (multiply-accumulate): nº de multiplicações+somas para classificar UM áudio. 1 MAC = 2 FLOPs.
  É o custo matemático da inferência, independente do hardware.

Cuidado (achado da POC): em eval() o Transformer do PyTorch usa um kernel fundido ("fast path") que o
contador de FLOPs não enxerga — subestima a Atenção em ~37x. Por isso contamos em modo train().
"""
import torch
from torch.utils.flop_counter import FlopCounterMode


def contar_parametros(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def contar_macs(model, n_mfcc=None, n_frames=None):
    """MACs por inferência de 1 amostra. O formato vem de model.formato_entrada (criar_modelo preenche)."""
    n_mfcc, n_frames = (n_mfcc, n_frames) if n_mfcc else getattr(model, "formato_entrada", (40, 101))
    estava_treinando = model.training
    model.train()   #desliga o fast path do Transformer (ver docstring); batch 2 por causa do BatchNorm
    with FlopCounterMode(display=False) as contador, torch.no_grad():
        model(torch.randn(2, n_mfcc, n_frames, device=next(model.parameters()).device))
    model.train(estava_treinando)
    return contador.get_total_flops() // 2 // 2     #/2 pelo batch, /2 de FLOPs para MACs


def custo(model, n_mfcc=None, n_frames=None):
    """Parâmetros, tamanho fp32/int8 (KB) e MACs por inferência."""
    params = contar_parametros(model)
    return dict(params=params, kb_fp32=params * 4 / 1024, kb_int8=params / 1024,
                macs=contar_macs(model, n_mfcc, n_frames))
