"""Avaliação fora do GSC: ruído real (DEMAND), AudioMNIST e MSWC-pt.

Todos usam o MESMO pipeline do treino: áudio -> 16 kHz mono, 1 s -> MFCC -> normalização (por amostra, ou com as
estatísticas do TREINO do GSC na global) -> modelo. Nenhum desses dados é visto no treino.

modelos: {nome: dict(model=..., extrator=..., media=..., desvio=...)} em todas as funções.
"""
import random

import torch

from dados.ruido import misturar_ruido
from dados.speech_commands import normalizar


@torch.no_grad()
def logits_de_wavs(model, wavs, extrator, media, desvio, batch=256):
    """Lista de waveforms (N_AMOSTRAS,) -> logits (len(wavs), n_classes).

    media/desvio = None quando a normalização é por amostra.
    """
    device = next(model.parameters()).device
    extrator = extrator.to(device)
    model.eval()
    saidas = []
    for i in range(0, len(wavs), batch):
        x = extrator(torch.stack(wavs[i:i + batch]).to(device))
        x = normalizar(x, None if media is None else media.to(device), None if desvio is None else desvio.to(device))
        saidas.append(model(x).cpu())
    return torch.cat(saidas)


def avaliar_ruido(modelos, wavs, y, ruidos, snrs, seed=0):
    """Acurácia de cada modelo em cada SNR. Devolve {nome: {'limpo'|snr: acc}}.

    modelos: {nome: dict(model=..., extrator=..., media=..., desvio=...)}. Cada modelo usa a SUA entrada
    (o MFCC e a normalização do seu HP_MODELOS[<modelo>]; a CNN 1D, por exemplo, usa 10 x 49).
    O MESMO trecho de ruído é usado para todos os modelos em cada SNR (comparação justa).
    """
    rng = random.Random(seed)
    res = {nome: {} for nome in modelos}
    for snr in ["limpo"] + list(snrs):
        if snr == "limpo":
            misturados = wavs
        else:
            misturados = [misturar_ruido(w, ruidos[rng.choice(list(ruidos))], snr, rng) for w in wavs]
        for nome, m in modelos.items():
            pred = logits_de_wavs(m["model"], misturados, m["extrator"], m["media"], m["desvio"]).argmax(1)
            res[nome][snr] = (pred == y).float().mean().item()
        print(snr, {n: f"{res[n][snr]*100:.1f}%" for n in res})
    return res


#---------------------------------------------------------------------------------------------------------------
#§10 do relatório (vindo do playground de generalização)
#---------------------------------------------------------------------------------------------------------------

def acuracia_por_modelo(modelos, wavs, y):
    """Acurácia (%) de cada modelo numa lista de áudios. Devolve {nome: acc}."""
    return {nome: (logits_de_wavs(m["model"], wavs, m["extrator"], m["media"], m["desvio"]).argmax(1) == y)
            .float().mean().item() * 100 for nome, m in modelos.items()}


def avaliar_ruido_por_ambiente(modelos, wavs, y, ruidos, snrs, seed=0):
    """Acurácia (%) com UM ambiente do DEMAND por vez. Devolve {snr: DataFrame (linhas: modelos, colunas: ambientes)}.

    Para cada ambiente e SNR, os mesmos áudios com o mesmo trecho de ruído vão para todos os modelos.
    """
    import pandas as pd
    por_ambiente = {snr: {} for snr in snrs}
    for amb, r in ruidos.items():
        rng = random.Random(seed)
        for snr in snrs:
            misturados = [misturar_ruido(w, r, snr, rng) for w in wavs]
            por_ambiente[snr][amb] = acuracia_por_modelo(modelos, misturados, y)
        print(amb, {snr: f"{sum(por_ambiente[snr][amb].values()) / len(modelos):.1f}%" for snr in snrs},
              "(média dos modelos)")
    return {snr: pd.DataFrame(d) for snr, d in por_ambiente.items()}


def previsoes_com_ruido(modelos, wavs, ruidos, snr, seed=0):
    """Mistura UMA vez (um ambiente sorteado por áudio) e devolve (previsões {nome: tensor}, ambiente de cada áudio)."""
    rng = random.Random(seed)
    amb_por_audio = [rng.choice(list(ruidos)) for _ in wavs]
    misturados = [misturar_ruido(w, ruidos[amb], snr, rng) for w, amb in zip(wavs, amb_por_audio)]
    preds = {nome: logits_de_wavs(m["model"], misturados, m["extrator"], m["media"], m["desvio"]).argmax(1)
             for nome, m in modelos.items()}
    return preds, amb_por_audio


def acuracia_por_ambiente(preds, y, amb_por_audio):
    """Acurácia (%) de cada modelo em cada ambiente (linhas: modelos, colunas: ambientes + 'média')."""
    import numpy as np
    import pandas as pd
    amb_t = np.array(amb_por_audio)
    ambientes = list(dict.fromkeys(amb_por_audio))
    acc = pd.DataFrame({amb: {nome: (pred[amb_t == amb] == y[amb_t == amb]).float().mean().item() * 100
                              for nome, pred in preds.items()}
                        for amb in sorted(ambientes)})
    acc["média"] = acc.mean(axis=1)
    return acc


def erros_por_ambiente(preds, y, amb_por_audio, classes):
    """Para onde vão os erros em cada ambiente. Devolve (tabela com todos os modelos juntos, tabela por modelo).

    A segunda tabela mostra, para cada modelo e ambiente, a resposta errada mais comum: se um modelo "colapsa"
    para a mesma palavra num ambiente, ela aparece ali com uma porcentagem alta.
    """
    import numpy as np
    import pandas as pd
    amb_t = np.array(amb_por_audio)
    ambientes = sorted(set(amb_por_audio))

    linhas = []
    for amb in ambientes:
        erros_prev, erros_conf = [], []
        for nome, pred in preds.items():
            sel = torch.from_numpy(amb_t == amb) & (pred != y)
            erros_prev += pred[sel].tolist()
            erros_conf += [(classes[v], classes[p]) for v, p in zip(y[sel].tolist(), pred[sel].tolist())]
        if not erros_prev:
            continue
        contagem = pd.Series([classes[p] for p in erros_prev]).value_counts()
        pares = pd.Series([f"{v} → {p}" for v, p in erros_conf]).value_counts()
        linhas.append({"ambiente": amb,
                       f"nº de erros ({len(preds)} modelos)": len(erros_prev),
                       "resposta errada mais comum": contagem.index[0],
                       "% dos erros com essa resposta": contagem.iloc[0] / len(erros_prev) * 100,
                       "3 respostas erradas mais comuns (% dos erros)":
                           ", ".join(f"{w} ({n / len(erros_prev) * 100:.0f}%)" for w, n in contagem.head(3).items()),
                       "troca mais comum: certa → respondida (vezes)": f"{pares.index[0]} ({pares.iloc[0]})"})
    tabela = pd.DataFrame(linhas).set_index("ambiente").sort_values("% dos erros com essa resposta", ascending=False)

    atrai = {}
    for amb in ambientes:
        atrai[amb] = {}
        for nome, pred in preds.items():
            erros = pred[torch.from_numpy(amb_t == amb) & (pred != y)]
            if len(erros) == 0:
                atrai[amb][nome] = "sem erros"
                continue
            palavra = erros.mode().values.item()
            atrai[amb][nome] = f"{classes[palavra]} ({(erros == palavra).float().mean().item() * 100:.0f}%)"
    return tabela, pd.DataFrame(atrai)


def com_ganho(wav, ganho_db):
    """O mesmo áudio mais alto ou mais baixo. clamp: o conversor A/D satura em ±1."""
    return (wav * 10 ** (ganho_db / 20)).clamp(-1, 1)


def avaliar_volume(modelos, wavs, y, ganhos_db):
    """Acurácia (%) com o áudio multiplicado por cada ganho. DataFrame (linhas: modelos, colunas: ganhos em dB)."""
    import pandas as pd
    res = {nome: {} for nome in modelos}
    for g in ganhos_db:
        accs = acuracia_por_modelo(modelos, [com_ganho(w, g) for w in wavs], y)
        for nome, acc in accs.items():
            res[nome][g] = acc
        print(f"{g:+d} dB", {n: f"{a:.1f}%" for n, a in accs.items()})
    return pd.DataFrame(res).T


def rms_db(wav):
    """Volume de um áudio em dB (RMS), só onde há sinal (o preenchimento de zeros até 1 s não conta)."""
    import math
    sinal = wav[wav.abs() > 1e-4]
    if len(sinal) == 0:
        return float("-inf")
    return 20 * math.log10(sinal.pow(2).mean().sqrt().item() + 1e-12)


def avaliar_digitos(logits, y_dig, idx_digitos):
    """AudioMNIST: acurácia (%) escolhendo entre as 35 palavras e só entre os 10 dígitos.

    logits: {nome: (N, 35)} | y_dig: dígito 0-9 de cada áudio | idx_digitos: posição de zero..nine nas 35 classes.
    """
    import pandas as pd
    return pd.DataFrame([
        {"modelo": nome,
         "acurácia 35 classes (%)": (L.argmax(1) == idx_digitos[y_dig]).float().mean().item() * 100,
         "acurácia só dígitos (%)": (L[:, idx_digitos].argmax(1) == y_dig).float().mean().item() * 100}
        for nome, L in logits.items()]).set_index("modelo")


def respostas_dos_modelos(logits, j, certo, classes, digitos, idx_digitos):
    """O que cada modelo respondeu para o áudio j (para a célula de ouvir). Devolve um DataFrame."""
    import pandas as pd
    linhas = []
    for nome, L in logits.items():
        prob = L[j].softmax(0)
        p35 = classes[prob.argmax().item()]
        pdig = digitos[prob[idx_digitos].argmax().item()]
        linhas.append({"modelo": nome,
                       "resposta (35 classes)": f"{p35} ({prob.max().item() * 100:.0f}%)",
                       "acertou (35)": "sim" if p35 == certo else "não",
                       "resposta (só dígitos)": pdig,
                       "acertou (dígitos)": "sim" if pdig == certo else "não"})
    return pd.DataFrame(linhas).set_index("modelo")


def acuracia_por_falante(logits, y_dig, idx_digitos, falantes, meta):
    """AudioMNIST: acurácia (%) de cada falante em cada modelo, com o grupo (nativo | com sotaque) e o sotaque."""
    import numpy as np
    import pandas as pd
    from dados.audiomnist import grupo_do_falante
    falantes_t = np.array(falantes)
    linhas = []
    for nome, L in logits.items():
        acerto35 = (L.argmax(1) == idx_digitos[y_dig]).float().numpy()
        acerto_dig = (L[:, idx_digitos].argmax(1) == y_dig).float().numpy()
        for f in sorted(set(falantes)):
            sel = falantes_t == f
            linhas.append({"modelo": nome, "falante": f, "grupo": grupo_do_falante(meta, f),
                           "sotaque": meta.get(f, {}).get("accent", "?"), "áudios": int(sel.sum()),
                           "acc 35 classes": acerto35[sel].mean() * 100,
                           "acc só dígitos": acerto_dig[sel].mean() * 100})
    return pd.DataFrame(linhas)


def resumo_nativos(por_falante):
    """Média ± desvio ENTRE falantes, por grupo, e a diferença nativo − com sotaque (p.p.), por modelo."""
    import pandas as pd
    resumo = por_falante.groupby(["modelo", "grupo"])[["acc 35 classes", "acc só dígitos"]].agg(["mean", "std"])
    modelos = list(dict.fromkeys(por_falante["modelo"]))
    tabela = pd.DataFrame(index=modelos)
    for medida in ["acc 35 classes", "acc só dígitos"]:
        for grupo in ["nativo", "com sotaque"]:
            tabela[f"{medida} | {grupo}"] = [f"{resumo.loc[(m, grupo), (medida, 'mean')]:.1f} ± "
                                             f"{resumo.loc[(m, grupo), (medida, 'std')]:.1f}" for m in modelos]
        tabela[f"{medida} | diferença (p.p.)"] = [resumo.loc[(m, "nativo"), (medida, "mean")]
                                                  - resumo.loc[(m, "com sotaque"), (medida, "mean")] for m in modelos]
    return tabela


def zero_shot_mapeado(modelos, wavs, y, idx_en):
    """MSWC-pt sem treino nenhum: a rede escolhe só entre as palavras em inglês equivalentes (sim -> yes ...).
    Devolve um DataFrame com a acurácia (%) e o acaso no nome da coluna."""
    import pandas as pd
    coluna = f"acurácia zero-shot (%) | acaso = {100 / len(idx_en):.1f}%"
    linhas = []
    for nome, m in modelos.items():
        L = logits_de_wavs(m["model"], wavs, m["extrator"], m["media"], m["desvio"])
        linhas.append({"modelo": nome, coluna: (L[:, idx_en].argmax(1) == y).float().mean().item() * 100})
    return pd.DataFrame(linhas).set_index("modelo")
