"""Loop de treino ÚNICO: todas as redes passam exatamente pelas mesmas regras.

Mesmo otimizador (AdamW), mesmo agendador (OneCycle), mesma perda (entropia cruzada com label smoothing),
mesmo aumento de dados. Só os hiperparâmetros de cada arquitetura mudam (configs/<modelo>.yaml).
Assim, diferenças de acurácia vêm da ARQUITETURA, não de truques de treino.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from configs import escolher_device, rotulo_experimento


def fixar_semente(seed):
    """Reprodutibilidade: mesma inicialização dos pesos e mesma ordem dos batches a cada execução."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def treinar_uma_epoca(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Uma passada pelo treino. Devolve a perda média."""
    model.train()   #ativa dropout e BatchNorm em modo treino
    soma_perdas, total_exemplos = 0.0, 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        #Zera os gradientes da iteração anterior: o PyTorch os ACUMULA
        optimizer.zero_grad()
        logits = model(x)                   #forward
        loss = criterion(logits, y)
        loss.backward()                     #backward: preenche os .grad
        optimizer.step()                    #atualiza os pesos
        if scheduler is not None:
            scheduler.step()                #OneCycle avança a cada batch, não a cada época

        soma_perdas += loss.item() * len(y)
        total_exemplos += len(y)

    return soma_perdas / total_exemplos


@torch.no_grad()
def validar_uma_epoca(model, dataloader, criterion, device):
    """Perda e acurácia no conjunto de validação. Devolve (perda, acurácia)."""
    model.eval()    #desliga dropout, BatchNorm usa as estatísticas acumuladas
    soma_perdas, acertos, total = 0.0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        soma_perdas += criterion(logits, y).item() * len(y)
        acertos += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return soma_perdas / total, acertos / total


@torch.no_grad()
def prever(model, dataloader, device=None):
    """Predições e rótulos verdadeiros de um conjunto inteiro. Devolve (y_pred, y_true) em CPU."""
    device = device or next(model.parameters()).device
    model.eval()
    preds, reais = [], []
    for x, y in dataloader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        reais.append(y)
    return torch.cat(preds), torch.cat(reais)


def _criar_otimizacao(model, hp, passos_totais):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler = None
    if hp.get("scheduler") == "onecycle":
        #Sobe o lr no começo (aquecimento) e desce suavemente até o fim
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hp["learning_rate"],
                                                        total_steps=passos_totais)
    return optimizer, scheduler


def treinar(model, loaders, hp, seed=0, pasta_runs=None, device=None, verbose=True):
    """Treina `model` e devolve (model com os MELHORES pesos na validação, hist).

    Se `pasta_runs` for dado, salva runs/<modelo>_s<seed>_<data>/ com config.yaml, hist.json e melhor.pt.
    """
    fixar_semente(seed)
    device = device or escolher_device()
    model = model.to(device)

    epocas = hp["epocas"]
    optimizer, scheduler = _criar_otimizacao(model, hp, epocas * len(loaders["treino"]))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.0))

    hist = {"treino": [], "val": [], "acc": [], "tempo_s": []}
    melhor_acc, melhor_estado = -1.0, None

    for epoca in range(1, epocas + 1):
        inicio = time.time()
        perda_treino = treinar_uma_epoca(model, loaders["treino"], criterion, optimizer, device, scheduler)
        perda_val, acc_val = validar_uma_epoca(model, loaders["val"], criterion, device)

        hist["treino"].append(perda_treino)
        hist["val"].append(perda_val)
        hist["acc"].append(acc_val)
        hist["tempo_s"].append(time.time() - inicio)

        #Guarda os pesos da melhor época (early stopping "passivo")
        if acc_val > melhor_acc:
            melhor_acc = acc_val
            melhor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"época {epoca:2d}/{epocas} | perda treino: {perda_treino:.4f} | "
                  f"perda val: {perda_val:.4f} | acurácia val: {acc_val*100:.2f}% | {hist['tempo_s'][-1]:.0f}s")

    model.load_state_dict(melhor_estado)
    if verbose:
        print(f"Melhor acurácia val: {melhor_acc*100:.2f}% (época {hist['acc'].index(melhor_acc)+1})")

    if pasta_runs is not None:
        _salvar_run(model, hp, hist, seed, pasta_runs)
    return model, hist


def _salvar_run(model, hp, hist, seed, pasta_runs):
    pasta = Path(pasta_runs) / f"{hp['modelo']}_{rotulo_experimento(hp)}_s{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    pasta.mkdir(parents=True, exist_ok=True)
    hist["pasta"] = str(pasta)
    yaml.safe_dump(dict(hp, seed=seed), open(pasta / "config.yaml", "w"), allow_unicode=True)
    json.dump(hist, open(pasta / "hist.json", "w"), indent=1)
    torch.save(model.state_dict(), pasta / "melhor.pt")
    return pasta


def carregar_run(pasta_runs, hp, seed):
    """Procura o run mais recente do mesmo modelo, tarefa, MFCC e seed. Devolve (state_dict, hist, hp) ou None."""
    candidatos = sorted(Path(pasta_runs).glob(f"{hp['modelo']}_{rotulo_experimento(hp)}_s{seed}_*"))
    if not candidatos:
        return None
    pasta = candidatos[-1]
    estado = torch.load(pasta / "melhor.pt", map_location="cpu")
    hist = json.load(open(pasta / "hist.json"))
    hp = yaml.safe_load(open(pasta / "config.yaml"))
    return estado, hist, hp
