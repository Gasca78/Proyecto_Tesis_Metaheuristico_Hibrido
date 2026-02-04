# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 11:29:10 2025

@author: oswal
"""
import opfunu
import config


# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
dims = config.DIMS

# ==========================================
# CARGA DE FUNCIONES (CEC 2027)
# ==========================================
# Nota: CEC2017 tiene funciones F1 a F29
# functions = [
#     opfunu.cec_based.cec2017.F12017(ndim=dims),
#     opfunu.cec_based.cec2017.F22017(ndim=dims),
#     opfunu.cec_based.cec2017.F32017(ndim=dims),
#     opfunu.cec_based.cec2017.F42017(ndim=dims),
#     opfunu.cec_based.cec2017.F52017(ndim=dims),
#     opfunu.cec_based.cec2017.F62017(ndim=dims),
#     opfunu.cec_based.cec2017.F72017(ndim=dims),
#     opfunu.cec_based.cec2017.F82017(ndim=dims),
#     opfunu.cec_based.cec2017.F92017(ndim=dims),
#     opfunu.cec_based.cec2017.F102017(ndim=dims),
#     opfunu.cec_based.cec2017.F112017(ndim=dims),
#     opfunu.cec_based.cec2017.F122017(ndim=dims),
#     opfunu.cec_based.cec2017.F132017(ndim=dims),
#     opfunu.cec_based.cec2017.F142017(ndim=dims),
#     opfunu.cec_based.cec2017.F152017(ndim=dims),
#     opfunu.cec_based.cec2017.F162017(ndim=dims),
#     opfunu.cec_based.cec2017.F172017(ndim=dims),
#     opfunu.cec_based.cec2017.F182017(ndim=dims),
#     opfunu.cec_based.cec2017.F192017(ndim=dims),
#     opfunu.cec_based.cec2017.F202017(ndim=dims),
#     opfunu.cec_based.cec2017.F212017(ndim=dims),
#     opfunu.cec_based.cec2017.F222017(ndim=dims),
#     opfunu.cec_based.cec2017.F232017(ndim=dims),
#     opfunu.cec_based.cec2017.F242017(ndim=dims),
#     opfunu.cec_based.cec2017.F252017(ndim=dims),
#     opfunu.cec_based.cec2017.F262017(ndim=dims),
#     opfunu.cec_based.cec2017.F272017(ndim=dims),
#     opfunu.cec_based.cec2017.F282017(ndim=dims),
#     opfunu.cec_based.cec2017.F292017(ndim=dims)
# ]

# # Funciones de Prueba para revisar resultados (resultados mostrados con probs fijas)
functions = [
    opfunu.cec_based.cec2017.F12017(ndim=dims), # Empate
    opfunu.cec_based.cec2017.F72017(ndim=dims), # No diferencias
    opfunu.cec_based.cec2017.F82017(ndim=dims), # Perdedor
    opfunu.cec_based.cec2017.F92017(ndim=dims), # Ganador
    opfunu.cec_based.cec2017.F262017(ndim=dims), # Perdedor (compleja)
    opfunu.cec_based.cec2017.F272017(ndim=dims), # Empate (compleja)
    opfunu.cec_based.cec2017.F282017(ndim=dims), # No diferencias
    opfunu.cec_based.cec2017.F292017(ndim=dims) # Ganador (compleja)
]

