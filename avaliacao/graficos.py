"""Gráficos padronizados (os mesmos para todos os modelos)."""
import matplotlib.pyplot as plt
import numpy as np


def curvas_treino(hist, titulo=""):
    """Perda (treino x val) e acurácia de validação por época."""
    eixo_x = range(1, len(hist["treino"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(eixo_x, hist["treino"], marker="o", label="treino")
    axes[0].plot(eixo_x, hist["val"], marker="s", label="validação")
    axes[0].set_title(f"Perda por época {titulo}")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Entropia cruzada")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(eixo_x, [a * 100 for a in hist["acc"]], marker="o", color="green")
    axes[1].set_title(f"Acurácia de validação {titulo}")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Acurácia (%)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_matriz_confusao(cm, classes, titulo="Matriz de confusão (normalizada por linha)"):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Prevista"); ax.set_ylabel("Verdadeira"); ax.set_title(titulo)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.show()


def acuracia_vs_custo(df, x="macs", y="acc_mean", rotulo="modelo", log_x=True):
    """Cada modelo é um ponto: quanto mais para cima e para a esquerda, melhor para a borda."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x], df[y] * 100, s=80)
    for _, r in df.iterrows():
        ax.annotate(r[rotulo], (r[x], r[y] * 100), textcoords="offset points", xytext=(6, 4))
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x); ax.set_ylabel("Acurácia (%)"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Acurácia × {x}")
    plt.tight_layout(); plt.show()


def barras_latencia(df):
    """Barras empilhadas: tempo do MFCC + tempo da rede, por modelo (hipótese b)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["modelo"], df["mfcc_ms"], label="MFCC (pré-processamento)")
    ax.bar(df["modelo"], df["rede_ms"], bottom=df["mfcc_ms"], label="rede (inferência)")
    ax.set_ylabel("ms por comando (batch=1, 1 thread)"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Onde está o custo: MFCC × rede")
    plt.tight_layout(); plt.show()


def curvas_ruido(res, snrs):
    """Acurácia por SNR, uma curva por modelo. res: {modelo: {snr: acc}}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rotulos = ["limpo"] + [f"{s} dB" for s in snrs]
    for nome, accs in res.items():
        ax.plot(rotulos, [accs[k] * 100 for k in ["limpo"] + list(snrs)], marker="o", label=nome)
    ax.set_xlabel("Relação sinal-ruído"); ax.set_ylabel("Acurácia (%)")
    ax.set_title("Queda de desempenho com ruído real (DEMAND)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def acuracia_vs_custo_varredura(resumo, margem_pp, referencia="baseline"):
    """Varredura de um modelo: acurácia de validação x custo, com o piso do critério de margem_pp p.p."""
    limite = resumo.loc[referencia, "acc_val"] - margem_pp
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, coluna, titulo in [(axes[0], "params", "parâmetros (tamanho)"), (axes[1], "macs", "MACs (computação)")]:
        ax.scatter(resumo[coluna], resumo["acc_val"])
        for nome, linha in resumo.iterrows():
            ax.annotate(nome, (linha[coluna], linha["acc_val"]), fontsize=7)
        ax.axhline(limite, linestyle="--", color="gray")     #o piso do critério de 2 p.p.
        ax.set_xscale("log")
        ax.set_xlabel(f"{titulo} (escala log)"); ax.set_ylabel("acurácia de validação (%)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def barras_preproc_rede(resumo):
    """Hipótese (b) na varredura: pré-processamento (MFCC + normalização) x rede, por configuração."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    posicao = np.arange(len(resumo))
    ax.bar(posicao - 0.2, resumo["mfcc_ms"] + resumo["norm_ms"], width=0.4, label="pré-processamento (MFCC + normalização)")
    ax.bar(posicao + 0.2, resumo["rede_ms"], width=0.4, label="rede")
    ax.set_xticks(posicao, resumo.index, rotation=45, ha="right")
    ax.set_ylabel("latência (ms, mediana)"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.show()


def plot_spec_augment(x, palavra, sorteios=3):
    """Um áudio (n_mfcc, n_frames) já normalizado: o original e `sorteios` sorteios do SpecAugment.
    Mesma escala de cor (vmin/vmax) em todos os gráficos."""
    from dados.speech_commands import spec_augment
    fig, axes = plt.subplots(1, sorteios + 1, figsize=(4 * (sorteios + 1), 3))
    axes[0].imshow(x, aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=3)
    axes[0].set_title(f"original: {palavra}")

    #sorteios diferentes do mesmo áudio
    for ax in axes[1:]:
        ax.imshow(spec_augment(x), aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=3)
        ax.set_title("com SpecAugment")
    plt.tight_layout(); plt.show()


def plot_batch(xb, yb, classes, n=8):
    """Para VER o que vai entrar na rede: os n primeiros MFCCs do batch, com a palavra de cada um."""
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(xb[i].cpu(), aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=3)
        ax.set_title(classes[yb[i]])
    plt.tight_layout(); plt.show()


#Microcontroladores de referência (KB). Fontes nos datasheets citados em docs/referencias.md
MICROCONTROLADORES = [
    dict(mcu="ATmega328P", placa="Arduino Uno", nucleo="AVR 8 bits, 16 MHz", flash_kb=31.5, sram_kb=2),     #32 KB - 0,5 KB do bootloader
    dict(mcu="MSP430FR5994", placa="LaunchPad MSP-EXP430FR5994", nucleo="MSP430 16 bits, 16 MHz", flash_kb=256, sram_kb=8),  #"Flash" = FRAM
    dict(mcu="nRF52840", placa="Arduino Nano 33 BLE Sense", nucleo="Cortex-M4F, 64 MHz", flash_kb=1024, sram_kb=256),
    dict(mcu="ESP32", placa="ESP32-WROOM-32", nucleo="Xtensa LX6 dual-core, 240 MHz", flash_kb=4096, sram_kb=520),  #Flash SPI externa do módulo
]


def cabe_em_microcontroladores(df, rotulo="modelo", mcus=MICROCONTROLADORES):
    """Os modelos (pesos e ativações em int8) contra a memória de microcontroladores conhecidos.

    Esquerda: Flash (onde ficam os pesos, `kb_int8`; `kb_fp32` em linha clara para comparação).
    Direita: SRAM (onde ficam as ativações durante a inferência, `kb_ram_int8`, estimativa).
    Cada barra é a capacidade de um MCU; cada linha vertical é um modelo: se a linha passa do fim da barra,
    o modelo não cabe. Retorna a tabela modelo x MCU com a fração da memória usada e se cabe.
    """
    import pandas as pd

    nomes = [f"{m['placa']}\n({m['mcu']})" for m in mcus]
    cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    paineis = [(axes[0], "flash_kb", "kb_int8", "Flash: pesos"), (axes[1], "sram_kb", "kb_ram_int8", "SRAM: ativações (estimativa)")]
    for ax, mem, col, titulo in paineis:
        ax.barh(nomes, [m[mem] for m in mcus], color="lightgray", edgecolor="gray")
        for i, m in enumerate(mcus):
            ax.text(m[mem], i, f" {m[mem]:g} KB", va="center", fontsize=8, color="dimgray")
        for k, (_, r) in enumerate(df.iterrows()):
            cor = cores[k % len(cores)]
            ax.axvline(r[col], color=cor, linestyle="--", linewidth=2, label=f"{r[rotulo]} int8 ({r[col]:.1f} KB)")
            if col == "kb_int8" and "kb_fp32" in df:
                ax.axvline(r["kb_fp32"], color=cor, linestyle=":", alpha=0.5, label=f"{r[rotulo]} fp32 ({r['kb_fp32']:.0f} KB)")
        ax.set_xscale("log")
        ax.set_xlim(1, max(m[mem] for m in mcus) * 4)
        ax.set_xlabel("KB (escala log)"); ax.set_title(titulo)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Em quais microcontroladores os modelos cabem (linha além da barra = não cabe)")
    plt.tight_layout(); plt.show()

    linhas = []
    for _, r in df.iterrows():
        for m in mcus:
            uso_flash, uso_sram = r["kb_int8"] / m["flash_kb"], r["kb_ram_int8"] / m["sram_kb"]
            linhas.append({rotulo: r[rotulo], "mcu": f"{m['placa']} ({m['mcu']})",
                           "flash usada (%)": round(100 * uso_flash, 1), "sram usada (%)": round(100 * uso_sram, 1),
                           "cabe": "sim" if uso_flash <= 1 and uso_sram <= 1 else "não"})
    return pd.DataFrame(linhas)
