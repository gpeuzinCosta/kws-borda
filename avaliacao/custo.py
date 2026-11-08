"""Custo computacional de um modelo — quanto ele "pesa" para um dispositivo restrito.

- Parâmetros: nº de pesos treináveis. Tamanho em memória = params x 4 bytes (fp32) ou x 1 byte (int8 quantizado).
- MACs (multiply-accumulate): nº de multiplicações+somas para classificar UM áudio. 1 MAC = 2 FLOPs.
  É o custo matemático da inferência, independente do hardware.

Cuidado: em eval() o Transformer do PyTorch usa um kernel fundido ("fast path") que o
contador de FLOPs não enxerga — subestima muito a Atenção. Por isso contamos em modo train().
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


def kb_ativacao_int8(model, n_mfcc=None, n_frames=None):
    """ESTIMATIVA da SRAM de pico para as ativações (KB, int8 = 1 byte por valor, batch 1).

    Passa 1 amostra pela rede e, em cada camada "folha", soma o tamanho da entrada e da saída (os dois buffers
    que precisam existir ao mesmo tempo); o pico é o maior desses valores. Não conta o buffer do MFCC/áudio,
    o overhead do runtime (ex.: TFLite Micro) nem tensores guardados para atalhos residuais, então o valor real
    no dispositivo é um pouco maior. Serve para comparar com a SRAM dos microcontroladores.
    """
    n_mfcc, n_frames = (n_mfcc, n_frames) if n_mfcc else getattr(model, "formato_entrada", (40, 101))
    pico = [0]

    def numel(t):
        if torch.is_tensor(t):
            return t.numel()
        if isinstance(t, (tuple, list)):
            return sum(numel(x) for x in t)
        return 0

    def gancho(_, entrada, saida):
        pico[0] = max(pico[0], numel(entrada) + numel(saida))

    folhas = [m for m in model.modules() if not list(m.children())]
    ganchos = [m.register_forward_hook(gancho) for m in folhas]
    estava_treinando = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.randn(1, n_mfcc, n_frames, device=next(model.parameters()).device))
    finally:
        for g in ganchos:
            g.remove()
        model.train(estava_treinando)
    return pico[0] / 1024


def custo(model, n_mfcc=None, n_frames=None):
    """Parâmetros, tamanho fp32/int8 (KB), SRAM estimada das ativações em int8 (KB) e MACs por inferência."""
    params = contar_parametros(model)
    return dict(params=params, kb_fp32=params * 4 / 1024, kb_int8=params / 1024,
                kb_ram_int8=kb_ativacao_int8(model, n_mfcc, n_frames),
                macs=contar_macs(model, n_mfcc, n_frames))
