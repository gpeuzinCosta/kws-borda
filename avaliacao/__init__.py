"""Avaliação (validação/teste): métricas, custo computacional, latência, registro de resultados e gráficos."""
from avaliacao.metricas import acuracia, f1_macro, matriz_confusao, top_confusoes, relatorio_por_classe
from avaliacao.custo import custo
from avaliacao.latencia import medir_latencia, cronometrar
from avaliacao.resultados import registrar, carregar_resultados
from avaliacao.externos import logits_de_wavs, avaliar_ruido
