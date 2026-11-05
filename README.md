# Reconhecimento de Comandos de Voz na Borda

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeuzinCosta/kws-borda/blob/main/notebooks/kws_borda_relatorio.ipynb)

Projeto final da disciplina de Deep Learning — **Guilherme Pulizzi, Jefferson Abraham, Luciano Pinheiro**.

Comparamos as famílias de redes vistas na disciplina (**MLP, CNN 1D, ResNet, DS-CNN e Atenção**) numa mesma
bancada de testes voltada para dispositivos restritos, no Google Speech Commands v2.

**Questões:**
- **(a)** É possível encolher essas arquiteturas clássicas e ainda manter um reconhecimento de voz funcional?
- **(b)** Em modelos tão pequenos, o que custa mais: o pré-processamento do áudio (MFCC) ou a inferência da rede?

> 📓 **Comece pelo notebook principal [`notebooks/kws_borda_relatorio.ipynb`](notebooks/kws_borda_relatorio.ipynb).**
> Ele conta o projeto inteiro, passo a passo, e reúne todos os resultados finais.

Prova de conceito anterior (que validou o pipeline e os datasets): [gpeuzinCosta/kws-borda-poc](https://github.com/gpeuzinCosta/kws-borda-poc).

---

## Estrutura

```
kws-borda/
├── notebooks/       📓 notebook principal (projeto + resultados) e experimentos/ (rascunhos)
├── modelos/         🧠 uma arquitetura por arquivo: mlp, cnn1d, resnet, dscnn, atencao
├── dados/           🎧 funções para gerir os datasets: download, MFCC, GSC v2, DEMAND, AudioMNIST, MSWC-pt
├── treino/          🏋️ loop de treino único (loop.py) + script de treino (treinar.py)
├── avaliacao/       📏 métricas, custo (params/MACs), latência, ruído, gráficos + script (avaliar.py)
├── configs/         ⚙️ hiperparâmetros: base.yaml (comum a todos) + <modelo>.yaml (só o que muda)
├── tests/           ✅ testes rápidos com dados falsos
└── docs/            📚 referências e achados da POC
```

| Pasta | Arquivos | Para que serve |
|---|---|---|
| `modelos/` | `__init__.py` (registro), `mlp.py`, `cnn1d.py`, `resnet.py`, `dscnn.py`, `atencao.py` | cada integrante implementa a sua rede aqui |
| `dados/` | `baixar.py`, `mfcc.py`, `speech_commands.py`, `ruido.py`, `audiomnist.py`, `mswc_pt.py` | baixar, ler e transformar o áudio em MFCC 40×101 |
| `treino/` | `loop.py`, `treinar.py` | o mesmo treino para todas as redes |
| `avaliacao/` | `metricas.py`, `custo.py`, `latencia.py`, `externos.py`, `resultados.py`, `graficos.py`, `avaliar.py` | acurácia, F1, matriz de confusão, MACs, latência MFCC × rede, ruído e generalização |
| `notebooks/` | `kws_borda_relatorio.ipynb`, `experimentos/template_modelo.ipynb` | narrativa do projeto + rascunhos |
| `configs/` | `base.yaml`, `<modelo>.yaml` | hiperparâmetros fora do código |

**Ideia central:** dados, pré-processamento, treino e avaliação são **iguais para todos**. Cada integrante só
implementa a sua arquitetura em `modelos/<modelo>.py`. Assim, a diferença nos resultados vem da arquitetura,
não de truques de treino.

## Como rodar no Colab (recomendado)

1. Clique no badge **Abrir no Colab** acima (o repositório é público, não precisa de token).
2. *Ambiente de execução → Alterar tipo de ambiente de execução → GPU* (Colab Pro: T4/L4/A100).
3. Rode a primeira célula (**Setup**). Ela monta o Google Drive, clona este repo em `/content/kws-borda` e
   instala as dependências.
4. *Ambiente de execução → Executar tudo*.

**Onde ficam os arquivos no Colab:**

| O quê | Onde | Por quê |
|---|---|---|
| `.tar.gz`/`.zip` dos datasets | `MyDrive/kws-borda/downloads/` | baixa **uma vez** só (o GSC tem 2,3 GB) |
| áudios extraídos | `/content/data/` (disco local da sessão) | ler 105 mil `.wav` direto do Drive é muito lento |
| cache de MFCC | `MyDrive/kws-borda/cache/` | não precisa recalcular a cada sessão |
| checkpoints + `resultados.csv` | `MyDrive/kws-borda/runs/` | não perde o treino se o runtime cair |

> Para o grupo compartilhar dados e treinos, uma pessoa cria `MyDrive/kws-borda` e compartilha a pasta. Os
> outros adicionam um atalho dela no próprio "Meu Drive".

**Alternativa: VS Code + Colab.** A extensão oficial *Google Colab* do VS Code conecta um notebook aberto
localmente a um runtime do Colab (inclusive o Pro). A célula de Setup funciona igual.

## Como rodar localmente

```bash
git clone https://github.com/gpeuzinCosta/kws-borda.git && cd kws-borda
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                                          # testes rápidos, sem baixar nada
python treino/treinar.py --modelo mlp --dados-falsos --epocas 1 # pipeline completo com dados sintéticos
python dados/baixar.py --dataset gsc                            # GSC v2 em data/gsc
python treino/treinar.py --modelo mlp                           # treino de verdade
python avaliacao/avaliar.py                                     # parâmetros, MACs e latência de todos
```

Sempre rode a partir da **raiz** do repositório.

## Como implementar um modelo

1. Abra `modelos/<modelo>.py`. O esqueleto já traz a arquitetura-alvo e o paper na docstring.
2. Implemente `__init__` e `forward` respeitando o contrato:
   - entrada `x`: `(B, 40, 101)` = MFCC normalizado (40 coeficientes × 101 janelas de 10 ms);
   - saída: logits `(B, n_classes)`;
   - a última camada se chama `self.head`.
3. Troque `IMPLEMENTADO = False` para `True`.
4. Ajuste os hiperparâmetros em `configs/<modelo>.yaml`.
5. `pytest` → `python treino/treinar.py --modelo <modelo> --dados-falsos --epocas 1`.
6. Abra um Pull Request (veja [Como colaborar](#como-colaborar)).

Para um modelo novo (fora dos 5): crie `modelos/<nome>.py` com `@registrar_modelo("<nome>")`, adicione o nome
à lista em `modelos/__init__.py` e crie `configs/<nome>.yaml`.

## Como colaborar

### Fluxo

1. Atualize o `main`: `git checkout main && git pull`.
2. Crie uma branch para o seu trabalho: `git checkout -b modelo/dscnn` (ou `dados/mswc`, `avaliacao/ruido`…).
3. Faça commits pequenos e com mensagens claras em português.
4. Antes do PR, rode `pytest`.
5. Abra um Pull Request para o `main` e peça revisão de outro integrante.

### Regras

- **Não altere o loop de treino, o pré-processamento nem as métricas por conta própria**: eles são a "régua"
  comum. Se precisar mudar algo, abra uma issue/PR separado e combine com o grupo, porque a mudança afeta
  todos os modelos.
- **Nunca versione dados, caches nem checkpoints** (`data/`, `cache/`, `runs/`, `*.pt`). Eles ficam no Drive.
- Hiperparâmetros vão no `configs/<modelo>.yaml`, não espalhados pelo código.
- **Notebooks:** os resultados finais vão para `notebooks/kws_borda_relatorio.ipynb`. Rascunhos vão para
  `notebooks/experimentos/<modelo>_<nome>.ipynb`, copiados do `template_modelo.ipynb`.
- Antes de commitar um notebook, limpe as saídas pesadas (*Editar → Limpar todas as saídas*), exceto no
  relatório final, quando ele for congelado para a entrega.

### Salvando um notebook editado no Colab

*Arquivo → Salvar uma cópia no GitHub* → escolha o repo `gpeuzinCosta/kws-borda`, a **sua branch** e o
mesmo caminho do arquivo. Depois abra o PR pelo GitHub.

### Divisão sugerida

| Frente | Responsável |
|---|---|
| Bancada (dados, treino, avaliação) + MLP | Guilherme |
| CNN 1D / ResNet | _a definir_ |
| DS-CNN / Atenção | _a definir_ |
| Ruído DEMAND, AudioMNIST, MSWC-pt | _a definir_ |

## Datasets

| Dataset | Papel no projeto | Observação (POC) |
|---|---|---|
| [Google Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands) | treino / validação / teste (35 palavras) | split oficial por locutor |
| [DEMAND (Zenodo 1227121)](https://zenodo.org/records/1227121) | **ruído real** misturado ao teste em SNR de 20 a −5 dB | não é fala: só ruído ambiente |
| [AudioMNIST](https://github.com/soerenab/AudioMNIST) | dígitos de outros falantes, sotaques e salas | mesmo microfone para todos |
| [MSWC português](https://mlcommons.org/datasets/multilingual-spoken-words/) | transferência de idioma (few-shot) | só ~14 palavras com clipes suficientes |

## Métricas

- Acurácia e F1 macro
- Matriz de confusão
- Parâmetros e tamanho (fp32/int8)
- MACs
- Latência em ms (batch = 1, 1 thread de CPU, mediana/p95), separando o **MFCC** da **rede**
