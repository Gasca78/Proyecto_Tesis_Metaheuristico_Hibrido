# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:53:12 2025

@author: oswal
"""

from mealpy import FloatVar
from mealpy import DE, PSO, GA
import numpy as np
from HIBRIDO import hibrid_JADE
import opfunu
import time

f = opfunu.name_based.a_func.Ackley01(30)

problem_dict = {
    "bounds":FloatVar(lb=f.lb, ub=f.ub),
    "minmax":"min",
    "obj_func":f.evaluate,
    "log_to":None
    }
# def objective_function(solution):
#      return np.sum(solution**2)

# problem_dict = {
#     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
#     "minmax": "min",
#     "obj_func": objective_function,
#     "log_to": None 
# }

hist_fitness = []
hist_times = []
rep = 10
epochs = 1000
pop_size = 100
for i in range(rep):
    # model = hibrid_JADE(epoch=epochs, pop_size=pop_size)
    # model = DE.OriginalDE(epoch=epochs, pop_size=pop_size)
    # model = PSO.OriginalPSO(epoch=epochs, pop_size=pop_size)
    # model = GA.BaseGA(epoch=epochs, pop_size=pop_size)
    # model = DE.JADE(epoch=epochs, pop_size=pop_size)
    # Tomamos tiempo al iniciar
    start_time = time.time()
    # Resolvemos
    g_best = model.solve(problem_dict)
    # Tomamos tiempo al finalizar
    end_time = time.time()
    # Obtenemos la duración de la ejecución
    execution_time = end_time - start_time
    fitness = model.g_best.target.fitness
    hist_fitness.append(fitness)
    hist_times.append(execution_time)
    print(f"Run {i+1}/{rep} -> Fitness: {fitness} | Tiempo: {execution_time:.4f} s")
# Reporte Final de Resultados 
print("\n" + "="*40)
print("       RESUMEN DE RESULTADOS")
print("="*40)
print(f"Modelo: {model}")
print(f"Función: Ackley01 (30 dims)")
print("-" * 40)
print(f"Mejor Fitness Encontrado : {np.min(hist_fitness)}")
print(f"Peor Fitness Encontrado  : {np.max(hist_fitness)}")
print(f"Fitness Promedio (Mean)  : {np.mean(hist_fitness)}")
print(f"Desviación Estándar (Std): {np.std(hist_fitness)}") # Importante para ver estabilidad
print("-" * 40)
print(f"Tiempo Total Acumulado   : {np.sum(hist_times):.4f} s")
print(f"Tiempo Promedio por Run  : {np.mean(hist_times):.4f} s")
print("="*40)