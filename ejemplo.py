# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:38:34 2026

@author: oswal
"""

from mealpy import FloatVar, DE, GA
from HIBRIDO import hibrid_JADE
import numpy as np
from benchmarks import problemas_ingenieria
from benchmarks.tsp import tsp
import time

# PRUEBA PARA PROBLEMAS DE INGENIERIA
# prob = problemas_de_ingenieria.problem

# PRUEBA PARA TSP (TRAVELING SALESMAN PROBLEM)
# prob = tsp.problem

problem = {
    "obj_func": prob.evaluate,
    "bounds": FloatVar(lb=prob.lb, ub=prob.ub),
    "minmax": "min",
    "log_to": None
}

# Correr poquitas épocas
model_JADE = DE.JADE(epoch=1000, pop_size=100)
model_HIB = hibrid_JADE(epoch=1000, pop_size=100)
model_GA = GA.BaseGA(epoch=1000, pop_size=100)
start_time_JADE = time.time()
g_best_JADE = model_JADE.solve(problem)
end_time_JADE = time.time()
start_time_HIB = time.time()
g_best_HIB = model_HIB.solve(problem)
end_time_HIB = time.time()
start_time_GA = time.time()
g_best_GA = model_GA.solve(problem)
end_time_GA = time.time()

print(f"Resultados problema: {prob.name}")
print(f"Best fitness Híbrido: {g_best_HIB.target.fitness}, Tiempo Híbrido: {(end_time_HIB-start_time_HIB):4f}")
print(f"Best fitness JADE: {g_best_JADE.target.fitness}, Tiempo JADE: {(end_time_JADE-start_time_JADE):4f}")
print(f"Best fitness GA: {g_best_GA.target.fitness}, Tiempo GA: {(end_time_GA-start_time_GA):4f}")




