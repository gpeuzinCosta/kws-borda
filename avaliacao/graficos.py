"""Gráficos padronizados (os mesmos para todos os modelos)."""
import matplotlib.pyplot as plt


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
