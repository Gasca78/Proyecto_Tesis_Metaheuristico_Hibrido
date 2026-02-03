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
# CARGA DE FUNCIONES (CEC 2020)
# ==========================================
# Nota: CEC2020 tiene funciones F1 a F10
functions = [
    opfunu.cec_based.cec2020.F12020(ndim=dims),
    opfunu.cec_based.cec2020.F22020(ndim=dims),
    opfunu.cec_based.cec2020.F32020(ndim=dims),
    opfunu.cec_based.cec2020.F42020(ndim=dims),
    opfunu.cec_based.cec2020.F52020(ndim=dims),
    opfunu.cec_based.cec2020.F62020(ndim=dims),
    opfunu.cec_based.cec2020.F72020(ndim=dims),
    opfunu.cec_based.cec2020.F82020(ndim=dims),
    opfunu.cec_based.cec2020.F92020(ndim=dims),
    opfunu.cec_based.cec2020.F102020(ndim=dims),
]

