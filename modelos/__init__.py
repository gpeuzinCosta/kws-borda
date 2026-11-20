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
    if hp.get("entrada") == "onda":
        #Áudio cru (B, 1, 16000): a rede da família vem atrás de uma frente que faz o papel do MFCC (modelos/onda.py)
        from modelos.onda import embrulhar_onda
        model = embrulhar_onda(cls, hp, extras)
    else:
        model = cls(n_classes=hp["n_classes"], n_mfcc=hp["n_mfcc"], n_frames=hp["n_frames"], **extras)
    #Guardado no modelo para a contagem de MACs saber o formato da entrada
    model.formato_entrada = (hp["n_mfcc"], hp["n_frames"])
    return model


#Hiperparâmetros da ENTRADA que mudam o formato do que chega na rede (e os pesos da frente, no áudio cru)
CHAVES_ENTRADA = ("entrada", "n_mfcc", "n_frames", "n_mels", "janela_ms", "passo_ms", "center", "n_classes",
                  "onda_frente", "onda_filtros", "onda_janela_ms", "onda_passo_ms", "onda_center")


def arquitetura(nome, hp):
    """O que define os pesos do modelo `nome`: os argumentos do __init__ (com o padrão quando faltam) e a entrada.

    Dois hp com a mesma arquitetura geram modelos com os mesmos tensores; serve para não carregar do Drive um treino
    de outra configuração (ex.: MLP de 128 neurônios no lugar do de 64).
    """
    _importar_todos()
    params = inspect.signature(MODELOS[nome].__init__).parameters
    arq = {k: hp.get(k, p.default) for k, p in params.items()
           if k != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    arq.update({k: hp.get(k) for k in CHAVES_ENTRADA})
    return arq


def esta_implementado(nome):
    _importar_todos()
    return getattr(MODELOS[nome], "IMPLEMENTADO", True)


def listar_modelos():
    """Tabela do registro: nome, família, referência, implementado?"""
    _importar_todos()
    return [dict(nome=n, familia=c.FAMILIA, referencia=c.REFERENCIA,
                 implementado=getattr(c, "IMPLEMENTADO", True)) for n, c in MODELOS.items()]
