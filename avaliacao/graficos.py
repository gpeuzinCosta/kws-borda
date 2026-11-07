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
