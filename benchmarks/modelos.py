# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:52:42 2026

@author: oswal
"""

from algoritmos import HIBRIDO, ETO, SPA 
from mealpy import DE, SHADE, PSO, GA, GWO, ACOR
from benchmarks.modelos_evaluados import PSO_Metrics, P_PSO_Metrics, JADE_Metrics, SADE_Metrics, GA_Metrics, ISLO_Metrics, ETO_Metrics, SPA_Metrics, SHADE_Metrics, GWO_Metrics

models = [
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int5_Falso',
        'parametros': {'update_interval': 5, 'matriz_type': 'moderado', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int10_Falso',
        'parametros': {'update_interval': 10, 'matriz_type':'moderado', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int50_Falso',
        'parametros': {'update_interval': 50, 'matriz_type':'moderado', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int5_Verdadero',
        'parametros': {'update_interval': 5, 'matriz_type': 'moderado', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int10_Verdadero',
        'parametros': {'update_interval': 10, 'matriz_type':'moderado', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int50_Verdadero',
        'parametros': {'update_interval': 50, 'matriz_type':'moderado', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Estricto_Int5_Falso',
        'parametros': {'update_interval': 5, 'matriz_type': 'estricto', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Estricto_Int10_Falso',
        'parametros': {'update_interval': 10, 'matriz_type':'estricto', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Estricto_Int50_Falso',
        'parametros': {'update_interval': 50, 'matriz_type':'estricto', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Estricto_Int5_Verdadero',
        'parametros': {'update_interval': 5, 'matriz_type': 'estricto', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Estricto_Int10_Verdadero',
        'parametros': {'update_interval': 10, 'matriz_type':'estricto', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Estricto_Int50_Verdadero',
        'parametros': {'update_interval': 50, 'matriz_type':'estricto', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Conservador_Int5_Falso',
        'parametros': {'update_interval': 5, 'matriz_type': 'conservador', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Conservador_Int10_Falso',
        'parametros': {'update_interval': 10, 'matriz_type':'conservador', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Conservador_Int50_Falso',
        'parametros': {'update_interval': 50, 'matriz_type':'conservador', 'success_filter': False, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Conservador_Int5_Verdadero',
        'parametros': {'update_interval': 5, 'matriz_type': 'conservador', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Conservador_Int10_Verdadero',
        'parametros': {'update_interval': 10, 'matriz_type':'conservador', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Conservador_Int50_Verdadero',
        'parametros': {'update_interval': 50, 'matriz_type':'conservador', 'success_filter': True, 'memory_type': 'markov'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Probabilidades_Int5_Falso',
        'parametros': {'update_interval': 5, 'success_filter': False, 'memory_type': 'probs'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Probabilidades_Int10_Falso',
        'parametros': {'update_interval': 10, 'success_filter': False, 'memory_type': 'probs'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Probabilidades_Int50_Falso',
        'parametros': {'update_interval': 50, 'success_filter': False, 'memory_type': 'probs'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Probabilidades_Int5_Verdadero',
        'parametros': {'update_interval': 5, 'success_filter': True, 'memory_type': 'probs'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Probabilidades_Int10_Verdadero',
        'parametros': {'update_interval': 10, 'success_filter': True, 'memory_type': 'probs'}
    },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Probabilidades_Int50_Verdadero',
        'parametros': {'update_interval': 50, 'success_filter': True, 'memory_type': 'probs'}
    }
]

# models_no_hybrids = [
#     {
#         'modelo': DE.JADE,
#         'nombre_carpeta': 'JADE',
#         'parametros': {}
#     },
#     {
#         'modelo': SHADE.OriginalSHADE,
#         'nombre_carpeta': 'SHADE',
#         'parametros': {}
#     },
#     {
#         'modelo': PSO.OriginalPSO,
#         'nombre_carpeta': 'PSO',
#         'parametros': {}
#     },
#     {
#         'modelo': GA.BaseGA,
#         'nombre_carpeta': 'GA',
#         'parametros': {}
#     },
#     {
#         'modelo': GWO.OriginalGWO,
#         'nombre_carpeta': 'GWO',
#         'parametros': {}
#     },
#     {
#         'modelo': ACOR.OriginalACOR(),
#         'nombre_carpeta': 'ACO',
#         'parametros': {}
#     }
# ]

# models = [ 
#     # {
#     #     'modelo': DE.JADE,
#     #     'nombre_carpeta': 'JADE',
#     #     'parametros': {}
#     # },
#     # {
#     #     'modelo': PSO.OriginalPSO,
#     #     'nombre_carpeta': 'PSO',
#     #     'parametros': {}
#     # },
#     # {
#     #     'modelo': GA.BaseGA,
#     #     'nombre_carpeta': 'GA',
#     #     'parametros': {}
#     # },
#     {
#         'modelo': HIBRIDO.hibrid_JADE,
#         'nombre_carpeta': 'Hibrido_Markov_Conservador_Int5_Verdadero',
#         'parametros': {'update_interval': 5, 'matriz_type': 'conservador', 'success_filter': True, 'memory_type': 'markov'}
#     },
#     {
#         'modelo': HIBRIDO.hibrid_JADE,
#         'nombre_carpeta': 'Hibrido_Probabilidades_Int5_Verdadero',
#         'parametros': {'update_interval': 5, 'success_filter': True, 'memory_type': 'probs'}
#     }
# ]
        
models_metrics = [
    {
        'modelo': JADE_Metrics,
        'nombre_carpeta': 'JADE',
        'parametros': {}
    },
    
    {
        'modelo': SADE_Metrics,
        'nombre_carpeta': 'SADE',
        'parametros': {}
    },
    {
        'modelo': PSO_Metrics,
        'nombre_carpeta': 'PSO',
        'parametros': {}
    },
    {
        'modelo': P_PSO_Metrics,
        'nombre_carpeta': 'P_PSO',
        'parametros': {}
    },
    {
        'modelo': GA_Metrics,
        'nombre_carpeta': 'GA',
        'parametros': {}
    },
    {
        'modelo': ISLO_Metrics,
        'nombre_carpeta': 'ILSO',
        'parametros': {}
    },
    {
        'modelo': ETO_Metrics,
        'nombre_carpeta': 'ETO',
        'parametros': {}
    },
    {
        'modelo': SPA_Metrics,
        'nombre_carpeta': 'SPA',
        'parametros': {}
    },
    # {
    #     'modelo': SHADE_Metrics,
    #     'nombre_carpeta': 'SHADE',
    #     'parametros': {}
    # },
    # {
    #     'modelo': GWO_Metrics,
    #     'nombre_carpeta': 'GWO',
    #     'parametros': {}
    # }
]
