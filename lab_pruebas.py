# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:38:34 2026

@author: oswal
"""

from mealpy import FloatVar, DE, GA
from HIBRIDO import hibrid_JADE
import numpy as np
from enoppy.paper_based import rwco_2020
import tsplib95
from benchmarks.mkp import mkp
import time

# # Problema dummy súper rápido
# # def obj_func(x): return np.sum(x**2)
# prob = rwco_2020.WeightMinimizationSpeedReducerProblem()
# PRUEBA PARA TSP (TRAVELING SALESMAN PROBLEM)

# prob = tsplib95.load('benchmarks/tsp/bays29.tsp')
# prob = tsplib95.load('benchmarks/tsp/burma14.tsp')
# prob = tsplib95.load('benchmarks/tsp/berlin52.tsp')
# prob = tsplib95.load('benchmarks/tsp/eil76.tsp')
problems = mkp.problems

# def fitness_tsp(solution):
#     # Obtenemos la ruta
#     ruta = np.argsort(solution)
#     # Calculamos la distancia total recorriendo la ruta
#     distancia_total = 0
#     for i in range(len(ruta)-1):
#         u = list(prob.get_nodes())[ruta[i]]
#         v = list(prob.get_nodes())[ruta[i+1]]
#         # La librería calcula la distancia (geográfica o euclidiana)
#         distancia_total += prob.get_weight(u, v)
#     # Sumar el regreso al inicio
#     u = list(prob.get_nodes())[ruta[-1]]
#     v = list(prob.get_nodes())[ruta[0]]
#     distancia_total += prob.get_weight(u, v)
    
#     return distancia_total

for prob in problems:
    problem = {
        "obj_func": prob.evaluate,
        "bounds": FloatVar(lb=prob.lb, ub=prob.ub),
        "minmax": "min",
        "log_to": None
    }
    # problem = {
    #     "obj_func": fitness_tsp,
    #     "bounds": FloatVar(lb=[0]*prob.dimension, ub=[1]*prob.dimension),
    #     "minmax": "min",
    #     "log_to": None
    # }
    
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
    # print(f"Best solution Híbrido: {g_best_HIB.solution}, Best fitness Híbrido: {g_best_HIB.target.fitness}, Time Híbrido: {end_time_HIB-start_time_HIB}")
    # print(f"Best solution JADE: {g_best_JADE.solution}, Best fitness JADE: {g_best_JADE.target.fitness}, Time JADE: {end_time_JADE-start_time_JADE}")
    # print(f"Best solution GA: {g_best_GA.solution}, Best fitness JADE: {g_best_GA.target.fitness}, Time GA: {end_time_GA-start_time_GA}")

# # --- LA PRUEBA DE FUEGO ---
# print("¿Tiene diversidad?:", hasattr(model.history, 'list_diversity'))
# print("¿Tiene exploración?:", hasattr(model.history, 'list_exploration'))

# # if hasattr(model.history, 'list_diversity'):
# #     print("Ejemplo de valores:", model.history.list_diversity[:3])
# ####################################################################################
# ####################################################################################

# from enoppy.paper_based.moeosma_2023 import SpeedReducerProblem
# # SRP = SpeedReducerProblem
# # SP = SpringProblem
# # HTBP = HydrostaticThrustBearingProblem
# # VPP = VibratingPlatformProblem
# # CSP = CarSideImpactProblem
# # WRMP = WaterResourceManagementProblem
# # BCP = BulkCarriersProblem
# # MPBPP = MultiProductBatchPlantProblem

# srp_prob = SpeedReducerProblem()
# print("Lower bound for this problem: ", srp_prob.lb)
# print("Upper bound for this problem: ", srp_prob.ub)
# x0 = srp_prob.create_solution()
# print("Get the objective values of x0: ", srp_prob.get_objs(x0))
# print("Get the constraint values of x0: ", srp_prob.get_cons(x0))
# print("Evaluate with default penalty function: ", srp_prob.evaluate(x0))

# ####################################################################################
# ####################################################################################



