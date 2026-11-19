"""Testes rápidos do pipeline com dados sintéticos (não baixam nada).

    pytest

Ao implementar um modelo, mude IMPLEMENTADO=True: o teste de contrato passa a rodar para ele.
"""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from configs import PRESETS_MFCC, carregar_config, montar_hp
from dados import (CLASSES_GSC, CLASSES_GSC12, ExtratorMFCC, criar_dataloaders, dados_falsos, ler_trecho,
                   listar_split, normalizar, rotulo, spec_augment_batch)
from avaliacao import acuracia, custo, f1_macro, matriz_confusao, medir_latencia, relatorio_por_classe
from modelos import MODELOS, criar_modelo, esta_implementado, listar_modelos, registrar_modelo
from treino import carregar_run, mesma_politica, prever, treinar, varredura

NOMES = [m["nome"] for m in listar_modelos()]


def test_config_mescla_base_e_modelo():
    hp = carregar_config("mlp", epocas=1)
    assert hp["modelo"] == "mlp" and hp["epocas"] == 1 and hp["n_mfcc"] == 10    #o MLP usa o MFCC hello_edge
    assert carregar_config("dscnn", epocas=1)["n_mfcc"] == 10 and carregar_config("atencao", epocas=1)["n_mfcc"] == 40
    assert hp["n_classes"] == 35 and carregar_config("mlp", tarefa="gsc12")["n_classes"] == 12


def test_montar_hp_igual_ao_carregar_config():
    #O relatório escreve os hiperparâmetros no notebook: com os valores de configs/__init__.py, tem que dar o mesmo dicionário
    from configs import HYPERPARAMETERS, HP_MODELOS
    hp = montar_hp(HYPERPARAMETERS, {"cnn1d": HP_MODELOS["cnn1d"]}, "cnn1d", epocas=3, fracao=None)
    assert hp == carregar_config("cnn1d", epocas=3)
    assert (hp["n_mfcc"], hp["n_frames"], hp["canais"]) == (10, 49, 48)


def test_carregar_config_nao_altera_os_padroes():
    hp = carregar_config("mlp")
    hp["seeds"].append(99)
    assert carregar_config("mlp")["seeds"] == [0, 1, 2]


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
    assert hist["melhor_epoca"] in (1, 2) and "gap" in hist
    assert carregar_run(tmp_path, hp, 0) is not None and carregar_run(tmp_path, hp, 0, nome_run="outro") is None


def test_early_stop_para_antes_do_teto():
    #Dados aleatórios: a validação não melhora de forma consistente, então a paciência estoura bem antes de 30 épocas
    hp = carregar_config("mlp", epocas=30, batch_size=32, tarefa="gsc12", paciencia=2, scheduler="plateau",
                         plateau_paciencia=1, min_delta=0.05)
    loaders, _ = criar_dataloaders(hp, dados_falsos(hp, n=64))
    model, hist = treinar(criar_modelo("mlp", hp), loaders, hp, seed=0, device=torch.device("cpu"), verbose=False)
    assert hist["parou_cedo"] and hist["epocas_treinadas"] == len(hist["acc"]) < 30
    assert hist["melhor_epoca"] <= hist["epocas_treinadas"] and len(hist["lr"]) == hist["epocas_treinadas"]
    assert mesma_politica(hp, hp) and not mesma_politica(dict(hp, paciencia=None), hp)
    #Um treino com a normalização global não serve para a por amostra (a varredura acha o run só pelo nome)
    assert not mesma_politica(dict(hp, normalizacao="global"), dict(hp, normalizacao="por_amostra"))


def test_latencia_separa_mfcc_e_rede():
    hp = carregar_config("mlp")
    media, desvio = torch.zeros(hp["n_mfcc"], 1), torch.ones(hp["n_mfcc"], 1)
    r = medir_latencia(criar_modelo("mlp", hp), hp, media, desvio, rodadas=2, repeticoes=5, aquecimento=1)
    assert r["mfcc_ms"] > 0 and r["norm_ms"] > 0 and r["rede_ms"] > 0
    assert r["preproc_ms"] == r["mfcc_ms"] + r["norm_ms"]


def test_spec_augment_batch():
    x = torch.randn(4096, 40, 101)
    x_original = x.clone()
    x_aum = spec_augment_batch(x)
    #Largura média das faixas: 5 de 101 janelas e 3,5 de 40 coeficientes -> ~13% de zeros
    assert 0.11 < (x_aum == 0).float().mean().item() < 0.16
    #Cada áudio sorteia as PRÓPRIAS faixas
    assert torch.unique((x_aum == 0).all(dim=1), dim=0).shape[0] > 100
    assert torch.equal(x, x_original)    #o batch original não muda


def test_dataloader_entrega_batch_pronto():
    hp = carregar_config("cnn1d", batch_size=32, fracao=0.5, normalizacao="global")
    loaders, (media, desvio) = criar_dataloaders(hp, dados_falsos(hp, n=128))
    xb, yb = next(iter(loaders["treino"]))
    assert xb.shape == (32, hp["n_mfcc"], hp["n_frames"]) and yb.shape == (32,)
    assert len(loaders["treino"].dataset) == 64          #metade do treino
    assert (xb == 0).float().mean() < 0.01               #o aumento NÃO acontece no DataLoader
    assert media.shape == (hp["n_mfcc"], 1)


def test_cnn1d_vencedora_nos_padroes():
    hp = carregar_config("cnn1d")
    assert (hp["n_mfcc"], hp["n_frames"], hp["canais"]) == (10, 49, 48)
    assert custo(criar_modelo("cnn1d", hp))["params"] == 24275     #o número da varredura do playground


def test_resnet_vencedora_nos_padroes():
    hp = carregar_config("resnet")
    assert (hp["n_mfcc"], hp["n_frames"]) == (10, 49)
    assert custo(criar_modelo("resnet", hp))["params"] == 43640     #res15-narrow, o número da varredura do playground


@pytest.mark.parametrize("nome,configs", [("mlp", varredura.CONFIGS_MLP), ("dscnn", varredura.CONFIGS_DSCNN),
                                           ("atencao", varredura.CONFIGS_ATENCAO)])
def test_configs_das_varreduras_do_relatorio(nome, configs):
    """As varreduras das §8.1, §8.4 e §8.5: nomes únicos, baseline primeiro e todo modelo cumpre o contrato."""
    nomes = [n for n, _ in configs]
    assert nomes[0] == "baseline" and len(set(nomes)) == len(nomes)
    hp = carregar_config(nome)
    for n, mudancas in configs:
        hp_exp = varredura.aplicar(hp, mudancas)
        model = criar_modelo(nome, hp_exp).eval()
        assert model(torch.randn(2, hp_exp["n_mfcc"], hp_exp["n_frames"])).shape == (2, hp_exp["n_classes"]), n


def test_modelo_do_notebook_substitui_o_esqueleto_e_roda_a_varredura(gsc_falso, tmp_path):
    """O caminho do playground: a classe escrita no notebook entra no registro no lugar do esqueleto."""
    listar_modelos()                 #garante que o esqueleto já está no registro
    original = MODELOS["resnet"]
    try:
        @registrar_modelo("resnet")
        class ResNetDeMentira(torch.nn.Module):
            def __init__(self, n_classes, n_mfcc=40, n_frames=101, canais=4):
                super().__init__()
                self.conv = torch.nn.Conv1d(n_mfcc, canais, 3, padding=1)
                self.head = torch.nn.Linear(canais, n_classes)

            def forward(self, x):
                return self.head(self.conv(x).mean(-1))

        listar_modelos()    #reimporta os arquivos de modelos/: não pode desfazer a troca
        assert MODELOS["resnet"] is ResNetDeMentira and esta_implementado("resnet")

        varredura.mfcc_em_memoria.update(nome=None, dados=None)
        hp = carregar_config("resnet", epocas=1, batch_size=16, **PRESETS_MFCC["hello_edge"])
        pastas = {"cache": tmp_path, "runs": tmp_path, "resultados": tmp_path}
        configs = [("baseline", {"canais": 4}), ("canais 2", {"canais": 2})]
        tabela, modelos = varredura.rodar_varredura("resnet", configs, hp, [0], gsc_falso, pastas,
                                                    device=torch.device("cpu"))
        assert list(tabela["experimento"]) == ["baseline", "canais 2"] and list(tabela["canais"]) == [4, 2]
        assert (tmp_path / "varredura_resnet.csv").exists()

        #Segunda chamada: carrega os runs salvos em vez de treinar de novo
        tabela2, modelos2 = varredura.rodar_varredura("resnet", configs, hp, [0], gsc_falso, pastas,
                                                      device=torch.device("cpu"))
        assert len(tabela2) == 2 and set(modelos2) == {("baseline", 0), ("canais 2", 0)}

        #Só uma parte das configurações (como no relatório): o CSV tem as duas, mas só a pedida volta
        tabela3, _ = varredura.rodar_varredura("resnet", configs[:1], hp, [0], gsc_falso, pastas,
                                               device=torch.device("cpu"))
        assert list(tabela3["experimento"]) == ["baseline"]

        latencias = {n: dict(mfcc_ms=1.0, norm_ms=0.1, rede_ms=0.5, preproc_ms=1.1, preproc_sobre_rede=2.2,
                             fator_x8=3.0, estavel=True, cpu="x") for n, _ in configs}
        resumo = varredura.montar_resumo(varredura.anexar_latencias(tabela, latencias), hp["fracao"], 100.0)
        assert varredura.escolher_vencedor(resumo) == "canais 2"    #com margem enorme, o menor vence
    finally:
        MODELOS["resnet"] = original


def test_split_oficial_nao_mistura_locutores(gsc_falso):
    from dados import locutores_em_comum
    assert locutores_em_comum(listar_split(gsc_falso, "gsc35")) == {"treino_e_teste": 0, "treino_e_val": 0}


def test_graficos_do_relatorio_rodam():
    import matplotlib
    matplotlib.use("Agg")
    from avaliacao.graficos import cabe_em_microcontroladores, plot_batch, plot_spec_augment
    plot_spec_augment(torch.randn(10, 49), "yes")
    plot_batch(torch.randn(8, 10, 49), torch.arange(8), CLASSES_GSC)
    import pandas as pd
    df = pd.DataFrame([dict(modelo="a", kb_fp32=40.0, kb_int8=10.0), dict(modelo="b", kb_fp32=400.0, kb_int8=100.0)])
    cabe = cabe_em_microcontroladores(df).set_index(["modelo", "mcu"])["cabe"]
    assert cabe["a", "Arduino Uno (ATmega328P)"] == "sim" and cabe["b", "Arduino Uno (ATmega328P)"] == "não"



#--------------------------------------------------------------------------------------------------------------
#Sem MFCC (§11) e generalização (§10)
#--------------------------------------------------------------------------------------------------------------

def test_normalizacao_por_amostra_e_o_padrao():
    from configs import HYPERPARAMETERS
    assert HYPERPARAMETERS["normalizacao"] == "por_amostra"
    loaders, (media, desvio) = criar_dataloaders(carregar_config("cnn1d"), dados_falsos(carregar_config("cnn1d"), n=32))
    assert media is None and desvio is None


@pytest.mark.parametrize("nome", NOMES)
def test_modelo_sem_mfcc_tem_a_mesma_rede(nome):
    """A frente aprendida entrega o MESMO formato do MFCC do modelo: a rede (e o custo dela) não muda."""
    from configs import sem_mfcc
    hp = carregar_config(nome)
    hp_onda = carregar_config(nome, **sem_mfcc(hp))
    assert (hp_onda["n_mfcc"], hp_onda["n_frames"]) == (1, 16000)
    model = criar_modelo(nome, hp_onda).eval()
    assert model(torch.randn(2, 1, 16000)).shape == (2, hp["n_classes"])
    assert model.formato_entrada == (1, 16000) and hasattr(model, "head")
    rede_mfcc = criar_modelo(nome, hp)
    #mesma rede: só os parâmetros da frente (conv sem bias + BatchNorm) a mais
    frente = sum(p.numel() for p in model.frente.parameters())
    assert custo(model)["params"] == custo(rede_mfcc)["params"] + frente
    assert frente == hp["n_mfcc"] * int(16 * hp["janela_ms"]) + 2 * hp["n_mfcc"]


def test_atencao_com_tokens_de_audio_cru():
    """A configuração 'áudio cru' da §8.5: 100 tokens de 10 ms (160 amostras), sem MFCC."""
    cfg = dict(varredura.CONFIGS_ATENCAO)["áudio cru (tokens de 10 ms)"]
    hp = varredura.aplicar(carregar_config("atencao"), cfg)
    model = criar_modelo("atencao", hp).eval()
    assert model.rede.projecao.in_features == 160 and model.rede.pos_embedding.shape[1] == 101
    x = torch.randn(3, 1, 16000) * 0.1
    assert model(x).shape == (3, 35)
    #A normalização do sinal fica dentro da rede: o volume não muda a saída
    assert torch.allclose(model(x), model(x * 3), atol=1e-3)


def test_treino_sem_mfcc_de_ponta_a_ponta(tmp_path):
    from configs import sem_mfcc
    hp = carregar_config("cnn1d", epocas=1, batch_size=16, tarefa="gsc12")
    hp = carregar_config("cnn1d", epocas=1, batch_size=16, tarefa="gsc12", **sem_mfcc(hp))
    dados = dados_falsos(hp, n=48)
    dados = {k: (X.half(), y) for k, (X, y) in dados.items()}      #o cache do áudio cru é float16
    loaders, (media, desvio) = criar_dataloaders(hp, dados)
    assert media is None and loaders["treino"].dataset.X.dtype == torch.float16
    model, hist = treinar(criar_modelo("cnn1d", hp), loaders, hp, seed=0, pasta_runs=tmp_path,
                          device=torch.device("cpu"), verbose=False)
    assert len(hist["acc"]) == 1 and carregar_run(tmp_path, hp, 0) is not None
    r = medir_latencia(model, hp, None, None, rodadas=1, repeticoes=3, aquecimento=1)
    assert r["rede_ms"] > 0


def test_few_shot_so_treina_a_head():
    from treino.few_shot import RedeCongelada, few_shot_congelado
    from configs import sem_mfcc
    for hp in [carregar_config("dscnn"), carregar_config("cnn1d", **sem_mfcc(carregar_config("cnn1d")))]:
        model = criar_modelo(hp["modelo"], hp)
        rede = RedeCongelada(model, 4)
        treinaveis = [n for n, p in rede.named_parameters() if p.requires_grad]
        assert treinaveis and all("head." in n for n in treinaveis)
        X = torch.randn(40, hp["n_mfcc"], hp["n_frames"])
        y = torch.arange(4).repeat(10)
        eh_treino = torch.arange(40) < 24
        media, desvio = few_shot_congelado(dict(model=model), X, y, eh_treino, K=3,
                                           hp_few=dict(epocas=2, batch_size=4, learning_rate=1e-2, weight_decay=0),
                                           repeticoes=2, device=torch.device("cpu"))
        assert 0 <= media <= 100
        assert model.head.out_features == 35       #o modelo original não muda
        #uma head só (peso e viés), e é a que o forward usa (sem uma cópia órfã no ModeloOnda)
        assert rede.rede.head.out_features == 4 and sum("head." in n for n, _ in rede.named_parameters()) == 2


def test_divisao_mswc_por_falante(tmp_path):
    import pandas as pd
    from dados.mswc_pt import dividir_por_falante
    linhas = []
    for w in ["sim", "não", "zero"]:
        for s in range(20):
            for k in range(4):
                rel = f"{w}/s{s}_{k}.opus"
                (tmp_path / w).mkdir(exist_ok=True)
                (tmp_path / rel).touch()
                linhas.append(dict(WORD=w, LINK=rel, SPEAKER=f"s{s}"))
    palavras, amostras = dividir_por_falante(pd.DataFrame(linhas), tmp_path, k_max=5, min_avaliacao=5)
    assert palavras == ["não", "sim", "zero"]
    treino = set(amostras.query("conjunto == 'treino'")["SPEAKER"])
    avaliacao = set(amostras.query("conjunto == 'avaliacao'")["SPEAKER"])
    assert treino and avaliacao and not (treino & avaliacao)
    assert amostras.groupby(["WORD", "SPEAKER"]).size().max() <= 3


def test_funcoes_da_generalizacao():
    import matplotlib
    matplotlib.use("Agg")
    from avaliacao import (com_ganho, avaliar_volume, previsoes_com_ruido, acuracia_por_ambiente,
                           erros_por_ambiente, avaliar_ruido_por_ambiente, zero_shot_mapeado)
    from avaliacao.graficos import curvas_volume, mapa_ambientes, comparar_treinos
    w = torch.rand(16000) * 2 - 1
    assert com_ganho(w, 20).abs().max() <= 1 and torch.allclose(com_ganho(w, 0), w)

    hp = carregar_config("cnn1d")
    modelos = {"cnn1d": dict(model=criar_modelo("cnn1d", hp).eval(), extrator=ExtratorMFCC.de_hparams(hp),
                             media=None, desvio=None)}
    wavs = [torch.randn(16000) * 0.1 for _ in range(6)]
    y = torch.zeros(6, dtype=torch.long)
    vol = avaliar_volume(modelos, wavs, y, [-10, 0])
    curvas_volume(vol)
    ruidos = {"A": torch.randn(32000), "B": torch.randn(32000)}
    por_amb = avaliar_ruido_por_ambiente(modelos, wavs, y, ruidos, [0])
    assert list(por_amb[0].columns) == ["A", "B"]
    preds, amb = previsoes_com_ruido(modelos, wavs, ruidos, -5)
    acc = acuracia_por_ambiente(preds, y, amb)
    mapa_ambientes(acc, -5)
    erros_por_ambiente(preds, y, amb, CLASSES_GSC)
    assert zero_shot_mapeado(modelos, wavs, y, torch.tensor([0, 1, 2])).shape == (1, 1)
    h = dict(acc=[0.1, 0.2], val=[2.0, 1.5], treino=[2.5, 1.8])
    comparar_treinos({"MFCC": h, "sem MFCC": h})


def test_varredura_nao_mistura_normalizacoes(gsc_falso, tmp_path):
    """O CSV no Drive tem a varredura global; refazer com a por amostra treina de novo e devolve só as linhas novas."""
    varredura.mfcc_em_memoria.update(nome=None, dados=None)
    pastas = {"cache": tmp_path, "runs": tmp_path, "resultados": tmp_path}
    configs = [("baseline", {}), ("norm global", {"normalizacao": "global"})]
    hp_global = carregar_config("mlp", epocas=1, batch_size=16, normalizacao="global")
    varredura.rodar_varredura("mlp", configs[:1], hp_global, [0], gsc_falso, pastas, device=torch.device("cpu"))

    hp = carregar_config("mlp", epocas=1, batch_size=16)
    tabela, modelos = varredura.rodar_varredura("mlp", configs, hp, [0], gsc_falso, pastas, device=torch.device("cpu"))
    assert list(zip(tabela["experimento"], tabela["normalizacao"])) == [("baseline", "por_amostra"), ("norm global", "global")]
    assert len(pd.read_csv(tmp_path / "varredura_mlp.csv")) == 3      #a linha global antiga continua no CSV
    #o baseline foi treinado de novo: o run mais recente é o da por amostra (o global não foi reaproveitado)
    import yaml
    run = sorted(tmp_path.glob("mlp_baseline_s0_*"))[-1]
    assert yaml.safe_load(open(run / "config.yaml"))["normalizacao"] == "por_amostra"
