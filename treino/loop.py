"""Loop de treino ÚNICO: todas as redes passam exatamente pelas mesmas regras.

Mesmo otimizador (AdamW), mesmo agendador (OneCycle), mesma perda (entropia cruzada com label smoothing),
mesmo aumento de dados (SpecAugment no batch, na GPU). Só os hiperparâmetros de cada arquitetura mudam
(configs/<modelo>.yaml). Assim, diferenças de acurácia vêm da ARQUITETURA, não de truques de treino.

treinar_uma_epoca e validar_uma_epoca vêm do playground da CNN 1D (notebooks/experimentos/).
"""
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from configs import escolher_device, rotulo_experimento
from dados.speech_commands import spec_augment_batch


def fixar_semente(seed):
    """Reprodutibilidade: mesma inicialização dos pesos e mesma ordem dos batches a cada execução."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def treinar_uma_epoca(model, dataloader, criterion, device,
                      optimizer, scheduler=None, aumento=None):

    model.train()   #ativa o dropout e o BatchNorm passa a usar as estatísticas do batch
    soma_perdas = 0.0
    total_exemplos = 0

    for mfccs, labels in dataloader:                       #images -> mfccs
        mfccs, labels = mfccs.to(device), labels.to(device)

        #Aumento de dados no batch inteiro, já na GPU (só no treino: o validar_uma_epoca não tem isso)
        if aumento is not None:
            mfccs = aumento(mfccs)

        #Zera os gradientes da iteração anterior
        optimizer.zero_grad()

        logits = model(mfccs)                  # forward
        loss = criterion(logits, labels)
        loss.backward()                        #backward: preenche os .grad

        #Atualiza os pesos.
        optimizer.step()

        #o OneCycle avança a cada BATCH, e não a cada época. Vem depois do optimizer.step()
        if scheduler is not None:
            scheduler.step()

        soma_perdas += loss.item() * mfccs.size(0)
        total_exemplos += mfccs.size(0)

    return soma_perdas / total_exemplos


@torch.no_grad()   #promete um foward mais rapido, não gravando as operações
def validar_uma_epoca(model, dataloader, criterion, device):
    model.eval()    #desliga o dropout e o BatchNorm passa a usar as estatísticas acumuladas no treino
    soma_perdas = 0.0
    acertos = 0
    total_exemplos = 0

    for mfccs, labels in dataloader:                       #images -> mfccs
        mfccs, labels = mfccs.to(device), labels.to(device)

        logits = model(mfccs)
        loss = criterion(logits, labels)

        preditos = logits.argmax(dim=1)   #a classe de maior logit
        acertos += (preditos == labels).sum().item()

        soma_perdas += loss.item() * mfccs.size(0)
        total_exemplos += mfccs.size(0)

    return soma_perdas / total_exemplos, acertos / total_exemplos


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
        #OneCycle: o lr começa pequeno, sobe até learning_rate e desce suave até quase zero no fim.
        #total_steps = batches por época x épocas, porque o scheduler.step() roda a cada BATCH
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hp["learning_rate"],
                                                        total_steps=passos_totais)
    return optimizer, scheduler


def treinar(model, loaders, hp, seed=0, pasta_runs=None, device=None, verbose=True, nome_run=None):
    """Treina `model` e devolve (model com os MELHORES pesos na validação, hist).

    Se `pasta_runs` for dado, salva runs/<modelo>_<nome_run ou rótulo do experimento>_s<seed>_<data>/ com
    config.yaml, hist.json e melhor.pt.
    """
    #Mesma semente = mesmos pesos iniciais e mesmos sorteios do SpecAugment
    fixar_semente(seed)
    device = device or escolher_device()
    model = model.to(device)

    epocas = hp["epocas"]
    optimizer, scheduler = _criar_otimizacao(model, hp, epocas * len(loaders["treino"]))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=hp.get("label_smoothing", 0.0))

    #O SpecAugment é aplicado no batch, dentro do treinar_uma_epoca
    if hp.get("spec_augment", False):
        aumento = spec_augment_batch
    else:
        aumento = None

    hist = {"treino": [], "val": [], "acc": [], "tempo_s": []}
    melhor_acc, melhor_estado = -1.0, None
    inicio = time.time()

    for epoca in range(1, epocas + 1):
        t0 = time.time()
        perda_treino = treinar_uma_epoca(model, loaders["treino"], criterion, device, optimizer, scheduler, aumento)
        perda_val, acc_val = validar_uma_epoca(model, loaders["val"], criterion, device)

        hist["treino"].append(perda_treino)
        hist["val"].append(perda_val)
        hist["acc"].append(acc_val)
        hist["tempo_s"].append(time.time() - t0)

        #Guarda os pesos da melhor época na validação.
        #O .clone() é essencial: o state_dict devolve referências aos pesos, que o treino continua alterando.
        #Sem a cópia, o "melhor estado" mudaria junto com o modelo
        if acc_val > melhor_acc:
            melhor_acc = acc_val
            melhor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"época {epoca:2d}/{epocas} | perda treino: {perda_treino:.4f} | "
                  f"perda val: {perda_val:.4f} | acurácia val: {acc_val*100:.2f}% | {hist['tempo_s'][-1]:.0f}s")

    model.load_state_dict(melhor_estado)   #devolve os pesos da melhor época, não os da última

    hist["melhor_acc"] = melhor_acc
    hist["melhor_epoca"] = hist["acc"].index(melhor_acc) + 1
    hist["acc_final"] = hist["acc"][-1]
    hist["gap"] = hist["val"][-1] - hist["treino"][-1]
    hist["tempo_min"] = (time.time() - inicio) / 60

    if verbose:
        print(f"Melhor acurácia val: {melhor_acc*100:.2f}% (época {hist['melhor_epoca']})")

    if pasta_runs is not None:
        _salvar_run(model, hp, hist, seed, pasta_runs, nome_run)
    return model, hist


def _prefixo_run(hp, nome_run):
    """Nome da pasta do run. Com nome_run (ex.: 'hello_edge + canais 48'), cada configuração da varredura tem
    a sua pasta: só o rótulo do experimento não distingue 'baseline' de 'sem weight_decay'."""
    if nome_run is None:
        return f"{hp['modelo']}_{rotulo_experimento(hp)}"
    slug = nome_run.replace("+", "mais").replace(" ", "_").replace("/", "-")
    return f"{hp['modelo']}_{slug}"


def _salvar_run(model, hp, hist, seed, pasta_runs, nome_run=None):
    pasta = Path(pasta_runs) / f"{_prefixo_run(hp, nome_run)}_s{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    pasta.mkdir(parents=True, exist_ok=True)
    hist["pasta"] = str(pasta)
    yaml.safe_dump(dict(hp, seed=seed), open(pasta / "config.yaml", "w"), allow_unicode=True)
    json.dump(hist, open(pasta / "hist.json", "w"), indent=1)
    torch.save(model.state_dict(), pasta / "melhor.pt")
    return pasta


def carregar_run(pasta_runs, hp, seed, nome_run=None):
    """Procura o run mais recente do mesmo modelo (e configuração) e seed. Devolve (state_dict, hist, hp) ou None."""
    candidatos = sorted(Path(pasta_runs).glob(f"{_prefixo_run(hp, nome_run)}_s{seed}_*"))
    if not candidatos:
        return None
    pasta = candidatos[-1]
    estado = torch.load(pasta / "melhor.pt", map_location="cpu")
    hist = json.load(open(pasta / "hist.json"))
    hp = yaml.safe_load(open(pasta / "config.yaml"))
    return estado, hist, hp
