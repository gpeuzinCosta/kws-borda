"""Registro de modelos: cada integrante pluga sua arquitetura sem mexer no treino nem na avaliação.

Contrato de TODO modelo:
    - classe torch.nn.Module decorada com @registrar_modelo("nome")
    - __init__(self, n_classes, n_mfcc=40, n_frames=101, **hp)   (hp = hiperparâmetros do YAML)
    - forward(x) com x: (B, n_mfcc, n_frames) = MFCC normalizado  ->  logits (B, n_classes)
      (n_mfcc/n_frames variam: 40x101 no MFCC padrão, 10x49 no preset hello_edge — NÃO fixe no código)

Uso:
    from modelos import criar_modelo, listar_modelos
    model = criar_modelo("dscnn", hp)

Num playground, escreva a classe no notebook com @registrar_modelo("resnet"): ela substitui o arquivo
modelos/resnet.py na sessão, e o treino, a varredura, a latência e os gráficos dos módulos passam a usá-la.
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

        #Um modelo definido FORA da pasta modelos/ (ex.: a ResNet escrita no playground) tem prioridade sobre o
        #arquivo do repositório: importar o esqueleto depois (o criar_modelo importa todos) não desfaz a troca
        atual = MODELOS.get(nome)
        if atual is not None and not atual.__module__.startswith("modelos.") and cls.__module__.startswith("modelos."):
            return cls

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
    aceitos = inspect.signature(cls.__init__).parameters
    extras = {k: v for k, v in hp.items() if k in aceitos and k not in ("n_classes", "n_mfcc", "n_frames")}
    model = cls(n_classes=hp["n_classes"], n_mfcc=hp["n_mfcc"], n_frames=hp["n_frames"], **extras)
    #Guardado no modelo para a contagem de MACs saber o formato da entrada
    model.formato_entrada = (hp["n_mfcc"], hp["n_frames"])
    return model


def esta_implementado(nome):
    _importar_todos()
    return getattr(MODELOS[nome], "IMPLEMENTADO", True)


def listar_modelos():
    """Tabela do registro: nome, família, referência, implementado?"""
    _importar_todos()
    return [dict(nome=n, familia=c.FAMILIA, referencia=c.REFERENCIA,
                 implementado=getattr(c, "IMPLEMENTADO", True)) for n, c in MODELOS.items()]
