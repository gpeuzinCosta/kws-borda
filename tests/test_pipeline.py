"""Testes rápidos do pipeline com dados sintéticos (não baixam nada).

    pytest

Ao implementar um modelo, mude IMPLEMENTADO=True: o teste de contrato passa a rodar para ele.
"""
import numpy as np
import pytest
import soundfile as sf
import torch

from configs import PRESETS_MFCC, carregar_config
from dados import (CLASSES_GSC, CLASSES_GSC12, ExtratorMFCC, criar_dataloaders, dados_falsos, ler_trecho,
                   listar_split, normalizar, rotulo)
from avaliacao import acuracia, custo, f1_macro, matriz_confusao, medir_latencia, relatorio_por_classe
from modelos import criar_modelo, esta_implementado, listar_modelos
from treino import prever, treinar

NOMES = [m["nome"] for m in listar_modelos()]


def test_config_mescla_base_e_modelo():
    hp = carregar_config("mlp", epocas=1)
    assert hp["modelo"] == "mlp" and hp["epocas"] == 1 and hp["n_mfcc"] == 40
    assert hp["n_classes"] == 35 and carregar_config("mlp", tarefa="gsc12")["n_classes"] == 12


@pytest.mark.parametrize("preset,formato", [("padrao", (40, 101)), ("hello_edge", (10, 49))])
def test_mfcc_tem_o_formato_do_preset(preset, formato):
    hp = carregar_config("mlp", **PRESETS_MFCC[preset])
    x = ExtratorMFCC.de_hparams(hp)(torch.randn(3, 16000))
    assert x.shape == (3, *formato) and (hp["n_mfcc"], hp["n_frames"]) == formato


@pytest.mark.parametrize("nome", NOMES)
@pytest.mark.parametrize("preset", list(PRESETS_MFCC))
def test_contrato_do_modelo(nome, preset):
    if not esta_implementado(nome):
        pytest.skip(f"{nome} ainda é esqueleto")
    hp = carregar_config(nome, **PRESETS_MFCC[preset])
    model = criar_modelo(nome, hp).eval()
    assert model(torch.randn(2, hp["n_mfcc"], hp["n_frames"])).shape == (2, hp["n_classes"])
    assert hasattr(model, "head"), "a última camada deve se chamar `head` (few-shot do MSWC)"
    c = custo(model)
    assert c["params"] > 0 and c["macs"] > 0


def test_normalizacao_por_amostra():
    x = normalizar(torch.randn(4, 10, 49) * 5 + 3)
    assert torch.allclose(x.mean(-1), torch.zeros(4, 10), atol=1e-5)


@pytest.fixture
def gsc_falso(tmp_path):
    """GSC em miniatura: 35 pastas com 6 áudios cada, listas oficiais e 2 ruídos de fundo."""
    val, teste = [], []
    for c in CLASSES_GSC:
        (tmp_path / c).mkdir()
        for i in range(6):
            sf.write(tmp_path / c / f"spk{i}_nohash_0.wav", np.random.randn(16000).astype("float32") * 0.1, 16000)
        val.append(f"{c}/spk4_nohash_0.wav")
        teste.append(f"{c}/spk5_nohash_0.wav")
    (tmp_path / "validation_list.txt").write_text("\n".join(val))
    (tmp_path / "testing_list.txt").write_text("\n".join(teste))
    (tmp_path / "_background_noise_").mkdir()
    for nome in ["white_noise.wav", "running_tap.wav"]:
        sf.write(tmp_path / "_background_noise_" / nome, np.random.randn(16000 * 5).astype("float32"), 16000)
    return tmp_path


def test_tarefa_12_classes(gsc_falso):
    split = listar_split(gsc_falso, "gsc12")
    #treino: 10 comandos x 4 = 40, + 4 unknown + 4 silence
    assert len(split["treino"]) == 48
    rotulos = {CLASSES_GSC12[rotulo(r, "gsc12")] for r in split["treino"]}
    assert rotulos == set(CLASSES_GSC12)
    silencio = next(r for r in split["treino"] if r.startswith("_silence_/"))
    assert ler_trecho(gsc_falso, silencio).shape == (16000,)
    #o sorteio é fixo: chamar de novo dá o mesmo split
    assert listar_split(gsc_falso, "gsc12") == split


def test_treino_e_avaliacao_de_ponta_a_ponta(tmp_path):
    hp = carregar_config("mlp", epocas=2, batch_size=32, tarefa="gsc12", normalizacao="por_amostra",
                         **PRESETS_MFCC["hello_edge"])
    loaders, (media, desvio) = criar_dataloaders(hp, dados_falsos(hp, n=128))
    assert media is None and desvio is None
    model, hist = treinar(criar_modelo("mlp", hp), loaders, hp, seed=0,
                          pasta_runs=tmp_path, device=torch.device("cpu"), verbose=False)
    assert len(hist["acc"]) == 2
    y_pred, y_true = prever(model, loaders["teste"])
    assert 0 <= acuracia(y_pred, y_true) <= 1 and 0 <= f1_macro(y_pred, y_true) <= 1
    assert matriz_confusao(y_pred, y_true, hp["n_classes"]).shape == (12, 12)
    assert len(relatorio_por_classe(y_pred, y_true, CLASSES_GSC12)) == 12
    assert (tmp_path / hist["pasta"].split("/")[-1] / "melhor.pt").exists()


def test_latencia_separa_mfcc_e_rede():
    hp = carregar_config("mlp")
    r = medir_latencia(criar_modelo("mlp", hp), repeticoes=5, aquecimento=1)
    assert r["mfcc_ms"] > 0 and r["rede_ms"] > 0
