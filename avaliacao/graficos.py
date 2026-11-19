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



#Microcontroladores de referência: Flash em KB. Fontes nos datasheets citados em docs/referencias.md
MICROCONTROLADORES = [
    dict(mcu="ATmega328P", placa="Arduino Uno", nucleo="AVR 8 bits, 16 MHz", flash_kb=31.5),     #32 KB - 0,5 KB do bootloader
    dict(mcu="MSP430FR5994", placa="LaunchPad MSP-EXP430FR5994", nucleo="MSP430 16 bits, 16 MHz", flash_kb=256),  #"Flash" = FRAM
    dict(mcu="nRF52840", placa="Arduino Nano 33 BLE Sense", nucleo="Cortex-M4F, 64 MHz", flash_kb=1024),
    dict(mcu="ESP32", placa="ESP32-WROOM-32", nucleo="Xtensa LX6 dual-core, 240 MHz", flash_kb=4096),  #Flash SPI externa do módulo
]


def cabe_em_microcontroladores(df, rotulo="modelo", mcus=MICROCONTROLADORES):
    """Os pesos dos modelos (`kb_int8`; `kb_fp32` em linha clara) contra a Flash de microcontroladores conhecidos.

    Cada barra é a Flash de um MCU; cada linha vertical é um modelo: se a linha passa do fim da barra, o modelo
    não cabe. Retorna a tabela modelo x MCU com a fração da Flash usada e se cabe.
    """
    import pandas as pd

    nomes = [f"{m['placa']}\n({m['mcu']})" for m in mcus]
    cores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(nomes, [m["flash_kb"] for m in mcus], color="lightgray", edgecolor="gray")
    for i, m in enumerate(mcus):
        ax.text(m["flash_kb"], i, f" {m['flash_kb']:g} KB", va="center", fontsize=8, color="dimgray")
    for k, (_, r) in enumerate(df.iterrows()):
        cor = cores[k % len(cores)]
        ax.axvline(r["kb_int8"], color=cor, linestyle="--", linewidth=2, label=f"{r[rotulo]} int8 ({r['kb_int8']:.1f} KB)")
        if "kb_fp32" in df:
            ax.axvline(r["kb_fp32"], color=cor, linestyle=":", alpha=0.5, label=f"{r[rotulo]} fp32 ({r['kb_fp32']:.0f} KB)")
    ax.set_xscale("log")
    ax.set_xlim(1, max(m["flash_kb"] for m in mcus) * 4)
    ax.set_xlabel("KB (escala log)")
    ax.set_title("Em quais microcontroladores os modelos cabem (linha além da barra = não cabe)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.show()

    linhas = []
    for _, r in df.iterrows():
        for m in mcus:
            uso = r["kb_int8"] / m["flash_kb"]
            linhas.append({rotulo: r[rotulo], "mcu": f"{m['placa']} ({m['mcu']})",
                           "flash usada (%)": round(100 * uso, 1), "cabe": "sim" if uso <= 1 else "não"})
    return pd.DataFrame(linhas)


#---------------------------------------------------------------------------------------------------------------
#§10 (generalização) e §11 (sem MFCC) do relatório
#---------------------------------------------------------------------------------------------------------------

def barras_por_ambiente(tabela, snr):
    """Acurácia por ambiente do DEMAND num SNR. tabela: DataFrame (linhas: modelos, colunas: ambientes)."""
    ax = tabela.T.plot.bar(figsize=(10, 4), ylim=(0, 100), ylabel="acurácia (%)",
                           title=f"Acurácia por ambiente do DEMAND ({snr} dB)")
    plt.xticks(rotation=0); ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()


def mapa_ambientes(acc_amb, snr):
    """Mapa de calor modelos x ambientes, do ambiente mais fácil para o mais difícil (a coluna 'média' fica de fora)."""
    ambientes = [c for c in acc_amb.columns if c != "média"]
    ordem = acc_amb[ambientes].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(acc_amb[ordem].values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ordem))); ax.set_xticklabels(ordem)
    ax.set_yticks(range(len(acc_amb))); ax.set_yticklabels(acc_amb.index)
    for i in range(len(acc_amb)):
        for j, amb in enumerate(ordem):
            v = acc_amb.loc[acc_amb.index[i], amb]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v < 60 else "black", fontsize=8)
    ax.set_title(f"Acurácia (%) por ambiente a {snr} dB (do mais fácil para o mais difícil)")
    fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()


def curvas_volume(tabela, marca_db=None, rotulo_marca=""):
    """Acurácia x ganho aplicado ao áudio, uma curva por modelo. tabela: linhas = modelos, colunas = ganhos (dB)."""
    plt.figure(figsize=(8, 5))
    for nome, linha in tabela.iterrows():
        plt.plot(list(tabela.columns), linha.values, marker="o", label=nome)
    if marca_db is not None:
        plt.axvline(marca_db, color="gray", linestyle="--", label=rotulo_marca)
    plt.xlabel("ganho aplicado ao áudio (dB)"); plt.ylabel("acurácia no teste do GSC (%)")
    plt.title("Robustez ao volume (GSC)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def falantes_nativos(por_falante):
    """AudioMNIST: cada ponto é um falante, os nativos em destaque (se ficam no meio da nuvem, o sotaque não explica)."""
    modelos = list(dict.fromkeys(por_falante["modelo"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    rng_jit = np.random.default_rng(0)
    for ax, medida in zip(axes, ["acc 35 classes", "acc só dígitos"]):
        for i, nome in enumerate(modelos):
            d = por_falante[por_falante["modelo"] == nome]
            for grupo, cor, tam in [("com sotaque", "tab:gray", 15), ("nativo", "tab:red", 60)]:
                g = d[d["grupo"] == grupo]
                ax.scatter(i + rng_jit.uniform(-0.15, 0.15, len(g)), g[medida], s=tam, color=cor, alpha=0.7,
                           label=grupo if i == 0 else None, zorder=3 if grupo == "nativo" else 2)
            ax.boxplot(d[medida], positions=[i], widths=0.5, showfliers=False)
        ax.set_xticks(range(len(modelos))); ax.set_xticklabels(modelos)
        ax.set_title(f"{medida}: acurácia por falante"); ax.set_ylabel("acurácia (%)"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    plt.tight_layout(); plt.show()


def curvas_few_shot(tabela, k_shots, acaso):
    """MSWC-pt: acurácia x K (exemplos por palavra), uma curva por modelo; o controle tracejado."""
    plt.figure(figsize=(8, 5))
    for nome, linha in tabela.iterrows():
        plt.plot(k_shots, [linha[f"_k{K}"] for K in k_shots], marker="o", label=nome,
                 linestyle="--" if nome.startswith("controle") else "-")
    plt.axhline(acaso, color="gray", linestyle=":", label="acaso")
    plt.xscale("log"); plt.xticks(k_shots, k_shots)
    plt.xlabel("exemplos de treino por palavra (K)"); plt.ylabel("acurácia nos falantes de avaliação (%)")
    plt.title("MSWC-pt: few-shot com a rede congelada (só a head treina)"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def comparar_treinos(hists, titulo=""):
    """Curvas de treino de várias versões do mesmo modelo, sobrepostas (ex.: com MFCC x sem MFCC).

    hists: {rótulo: hist}. Três painéis: acurácia de validação, perda de validação e perda de treino por época.
    O marcador maior é a melhor época (a que o early stop guardou).
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for rotulo, h in hists.items():
        x = range(1, len(h["acc"]) + 1)
        linha, = axes[0].plot(x, [a * 100 for a in h["acc"]], label=rotulo)
        melhor = int(np.argmax(h["acc"]))
        axes[0].plot(melhor + 1, h["acc"][melhor] * 100, "o", color=linha.get_color(), markersize=8)
        axes[1].plot(x, h["val"], color=linha.get_color(), label=rotulo)
        axes[2].plot(x, h["treino"], color=linha.get_color(), label=rotulo)
    for ax, nome in zip(axes, ["acurácia de validação (%)", "perda de validação", "perda de treino"]):
        ax.set_title(f"{nome} {titulo}"); ax.set_xlabel("época"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
