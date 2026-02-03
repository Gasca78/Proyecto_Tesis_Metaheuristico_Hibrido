# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:38:34 2026

@author: oswal
"""

from mealpy import FloatVar, DE
from HIBRIDO import hibrid_JADE
import numpy as np

# Problema dummy súper rápido
def obj_func(x): return np.sum(x**2)

problem = {
    "obj_func": obj_func,
    "bounds": FloatVar(lb=[-10]*5, ub=[10]*5),
    "minmax": "min",
    "log_to": None
}

# Correr poquitas épocas
# model = DE.JADE(epoch=10, pop_size=20)
model = hibrid_JADE(epoch=10, pop_size=20)
model.solve(problem)

# --- LA PRUEBA DE FUEGO ---
print("¿Tiene diversidad?:", hasattr(model.history, 'list_diversity'))
print("¿Tiene exploración?:", hasattr(model.history, 'list_exploration'))

if hasattr(model.history, 'list_diversity'):
    print("Ejemplo de valores:", model.history.list_diversity[:3])