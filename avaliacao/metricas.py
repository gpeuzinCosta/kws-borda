"""Métricas de desempenho.

- Acurácia: fração de áudios classificados corretamente.
- F1 macro: média do F1 de cada classe (F1 = média harmônica de precisão e revocação). Dá o mesmo peso a
  todas as palavras, então uma classe difícil ("forward", "learn") não some atrás das fáceis.
- Matriz de confusão: linha = palavra verdadeira, coluna = palavra prevista. Fora da diagonal estão os
  erros — e eles são FONÉTICOS (POC: forward->four, tree<->three, go<->no, off->up).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def _np(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def acuracia(y_pred, y_true):
    return float((_np(y_pred) == _np(y_true)).mean())


def f1_macro(y_pred, y_true):
    return float(f1_score(_np(y_true), _np(y_pred), average="macro"))


def matriz_confusao(y_pred, y_true, n_classes, normalizar=True):
    """Normalizada por linha: cada linha soma 1 (fração de cada palavra verdadeira)."""
    cm = confusion_matrix(_np(y_true), _np(y_pred), labels=list(range(n_classes))).astype(float)
    if normalizar:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm


def top_confusoes(cm, classes, k=10):
    """Os k pares (verdadeira -> prevista) mais confundidos, fora da diagonal."""
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm, axis=None)[:k], cm.shape))[0]
    return [(classes[i], classes[j], float(cm[i, j])) for i, j in idx]


def relatorio_por_classe(y_pred, y_true, classes):
    """Precisão, revocação, F1 e nº de exemplos de CADA palavra (trazido do notebook do Luciano).

    - precisão: das vezes que o modelo disse "go", quantas eram "go"?
    - revocação (recall): dos "go" verdadeiros, quantos o modelo achou?
    Ordenado do pior F1 para o melhor: as primeiras linhas mostram onde o modelo sofre.
    """
    r = classification_report(_np(y_true), _np(y_pred), labels=list(range(len(classes))),
                              target_names=classes, output_dict=True, zero_division=0)
    df = pd.DataFrame({c: r[c] for c in classes}).T
    df.columns = ["precisao", "revocacao", "f1", "exemplos"]
    df["exemplos"] = df["exemplos"].astype(int)
    return df.sort_values("f1")
