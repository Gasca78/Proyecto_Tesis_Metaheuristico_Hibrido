# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:52:42 2026

@author: oswal
"""

import HIBRIDO
from mealpy import DE, SHADE, PSO, GA, GWO, ACOR

models = [
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Moderado_Int5_Falso',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'moderado', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Moderado_Int10_Falso',
    #     'parametros': {'update_interval': 10, 'matriz_type':'moderado', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Moderado_Int50_Falso',
    #     'parametros': {'update_interval': 50, 'matriz_type':'moderado', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Moderado_Int5_Verdadero',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'moderado', 'success_filter': True, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Moderado_Int10_Verdadero',
    #     'parametros': {'update_interval': 10, 'matriz_type':'moderado', 'success_filter': True, 'memory_type': 'markov'}
    # },
    {
        'modelo': HIBRIDO.hibrid_JADE,
        'nombre_carpeta': 'Hibrido_Markov_Moderado_Int50_Verdadero',
        'parametros': {'update_interval': 50, 'matriz_type':'moderado', 'success_filter': True, 'memory_type': 'markov'}
    },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int5_Falso',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'extricto', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int10_Falso',
    #     'parametros': {'update_interval': 10, 'matriz_type':'extricto', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int50_Falso',
    #     'parametros': {'update_interval': 50, 'matriz_type':'extricto', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int5_Verdadero',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'extricto', 'success_filter': True, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int10_Verdadero',
    #     'parametros': {'update_interval': 10, 'matriz_type':'extricto', 'success_filter': True, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int50_Verdadero',
    #     'parametros': {'update_interval': 50, 'matriz_type':'extricto', 'success_filter': True, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Conservador_Int5_Falso',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'conservador', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Conservador_Int10_Falso',
    #     'parametros': {'update_interval': 10, 'matriz_type':'conservador', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Conservador_Int50_Falso',
    #     'parametros': {'update_interval': 50, 'matriz_type':'conservador', 'success_filter': False, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Conservador_Int5_Verdadero',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'conservador', 'success_filter': True, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Conservador_Int10_Verdadero',
    #     'parametros': {'update_interval': 10, 'matriz_type':'conservador', 'success_filter': True, 'memory_type': 'markov'}
    # },
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Conservador_Int50_Verdadero',
    #     'parametros': {'update_interval': 50, 'matriz_type':'conservador', 'success_filter': True, 'memory_type': 'markov'}
    # },
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
    # {
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Probabilidades_Int5_Verdadero',
    #     'parametros': {'update_interval': 5, 'success_filter': True, 'memory_type': 'probs'}
    # },
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

models_no_hybrids = [
    {
        'modelo': DE.JADE,
        'nombre_carpeta': 'JADE',
        'parametros': {}
    },
    {
        'modelo': SHADE.OriginalSHADE,
        'nombre_carpeta': 'SHADE',
        'parametros': {}
    },
    {
        'modelo': PSO.OriginalPSO,
        'nombre_carpeta': 'PSO',
        'parametros': {}
    },
    {
        'modelo': GA.BaseGA,
        'nombre_carpeta': 'GA',
        'parametros': {}
    },
    {
        'modelo': GWO.OriginalGWO,
        'nombre_carpeta': 'GWO',
        'parametros': {}
    }
    # {
    #     'modelo': ACOR.OriginalACOR(),
    #     'nombre_carpeta': 'ACO',
    #     'parametros': {}
    # }
]

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
