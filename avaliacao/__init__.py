"""Avaliação (validação/teste): métricas, custo computacional, latência, registro de resultados e gráficos."""
from avaliacao.metricas import acuracia, f1_macro, matriz_confusao, top_confusoes, relatorio_por_classe
from avaliacao.custo import custo
from avaliacao.latencia import medir_latencia, cronometrar
from avaliacao.resultados import registrar, carregar_resultados
from avaliacao.externos import (logits_de_wavs, avaliar_ruido, avaliar_ruido_por_ambiente, previsoes_com_ruido,
                                acuracia_por_ambiente, erros_por_ambiente, com_ganho, avaliar_volume, rms_db,
                                avaliar_digitos, respostas_dos_modelos, acuracia_por_falante, resumo_nativos,
                                zero_shot_mapeado)
