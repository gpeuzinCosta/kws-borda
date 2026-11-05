"""Registro de modelos: cada integrante pluga sua arquitetura sem mexer no treino nem na avaliação.

Contrato de TODO modelo:
    - classe torch.nn.Module decorada com @registrar_modelo("nome")
    - __init__(self, n_classes, n_mfcc=40, n_frames=101, **hp)   (hp = hiperparâmetros do YAML)
    - forward(x) com x: (B, n_mfcc, n_frames) = MFCC normalizado  ->  logits (B, n_classes)

Uso:
    from modelos import criar_modelo, listar_modelos
    model = criar_modelo("dscnn", hp)
"""
import importlib
import inspect

MODELOS = {}


def registrar_modelo(nome, familia=None, referencia=None):
    """Decorator que adiciona a classe ao registro."""
    def decorar(cls):
        cls.NOME = nome
        cls.FAMILIA = familia or nome
        cls.REFERENCIA = referencia
        MODELOS[nome] = cls
        return cls
    return decorar


def _importar_todos():
    #Importar os arquivos executa os decorators e preenche MODELOS
    for modulo in ["mlp", "cnn1d", "resnet", "dscnn", "atencao"]:
        importlib.import_module(f"modelos.{modulo}")


def criar_modelo(nome, hp):
    """Instancia o modelo `nome` passando só os hiperparâmetros que o __init__ dele aceita."""
    _importar_todos()
    if nome not in MODELOS:
        raise KeyError(f"Modelo '{nome}' não registrado. Disponíveis: {list(MODELOS)}")
    cls = MODELOS[nome]
    n_frames = int(hp["duracao_s"] * 1000 / hp["passo_ms"]) + 1
    aceitos = inspect.signature(cls.__init__).parameters
    extras = {k: v for k, v in hp.items() if k in aceitos and k not in ("n_classes", "n_mfcc", "n_frames")}
    return cls(n_classes=hp["n_classes"], n_mfcc=hp["n_mfcc"], n_frames=n_frames, **extras)


def esta_implementado(nome):
    _importar_todos()
    return getattr(MODELOS[nome], "IMPLEMENTADO", True)


def listar_modelos():
    """Tabela do registro: nome, família, referência, implementado?"""
    _importar_todos()
    return [dict(nome=n, familia=c.FAMILIA, referencia=c.REFERENCIA,
                 implementado=getattr(c, "IMPLEMENTADO", True)) for n, c in MODELOS.items()]
