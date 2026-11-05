"""Testes rápidos do pipeline com dados sintéticos (não baixam nada).

    pytest

Ao implementar um modelo, mude IMPLEMENTADO=True: o teste de contrato passa a rodar para ele.
"""
import pytest
import torch

from configs import carregar_config
from dados import ExtratorMFCC, criar_dataloaders, dados_falsos
from avaliacao import acuracia, custo, f1_macro, matriz_confusao, medir_latencia
from modelos import criar_modelo, esta_implementado, listar_modelos
from treino import prever, treinar

NOMES = [m["nome"] for m in listar_modelos()]


def test_config_mescla_base_e_modelo():
    hp = carregar_config("mlp", epocas=1)
    assert hp["modelo"] == "mlp" and hp["epocas"] == 1 and hp["n_mfcc"] == 40


def test_mfcc_tem_formato_40x101():
    hp = carregar_config("mlp")
    x = ExtratorMFCC.de_hparams(hp)(torch.randn(3, 16000))
    assert x.shape == (3, 40, 101)


@pytest.mark.parametrize("nome", NOMES)
def test_contrato_do_modelo(nome):
    if not esta_implementado(nome):
        pytest.skip(f"{nome} ainda é esqueleto")
    hp = carregar_config(nome)
    model = criar_modelo(nome, hp).eval()
    assert model(torch.randn(2, 40, 101)).shape == (2, hp["n_classes"])
    assert hasattr(model, "head"), "a última camada deve se chamar `head` (few-shot do MSWC)"
    c = custo(model)
    assert c["params"] > 0 and c["macs"] > 0


def test_treino_e_avaliacao_de_ponta_a_ponta(tmp_path):
    hp = carregar_config("mlp", epocas=2, batch_size=32, num_workers=0)
    loaders, _ = criar_dataloaders(hp, dados_falsos(hp, n=128))
    model, hist = treinar(criar_modelo("mlp", hp), loaders, hp, seed=0,
                          pasta_runs=tmp_path, device=torch.device("cpu"), verbose=False)
    assert len(hist["acc"]) == 2
    y_pred, y_true = prever(model, loaders["teste"])
    assert 0 <= acuracia(y_pred, y_true) <= 1 and 0 <= f1_macro(y_pred, y_true) <= 1
    assert matriz_confusao(y_pred, y_true, hp["n_classes"]).shape == (35, 35)
    assert (tmp_path / hist["pasta"].split("/")[-1] / "melhor.pt").exists()


def test_latencia_separa_mfcc_e_rede():
    hp = carregar_config("mlp")
    r = medir_latencia(criar_modelo("mlp", hp), repeticoes=5, aquecimento=1)
    assert r["mfcc_ms"] > 0 and r["rede_ms"] > 0
