# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:38:34 2026

@author: oswal
"""

from mealpy import FloatVar, DE
from HIBRIDO import hibrid_JADE
# import numpy as np
import enoppy

# # Problema dummy súper rápido
# def obj_func(x): return np.sum(x**2)
prob = enoppy.paper_based.rwco_2020.WeightMinimizationSpeedReducerProblem()

problem = {
    "obj_func": prob.evaluate,
    "bounds": FloatVar(lb=prob.lb, ub=prob.ub),
    "minmax": "min",
    "log_to": None
}

# Correr poquitas épocas
model_JADE = DE.JADE(epoch=1000, pop_size=100)
model_HIB = hibrid_JADE(epoch=1000, pop_size=100)

g_best_JADE = model_JADE.solve(problem)
g_best_HIB = model_HIB.solve(problem)

print(f"Best solution JADE: {g_best_JADE.solution}, Best fitness JADE: {g_best_JADE.target.fitness}")
print(f"Best solution Híbrido: {g_best_HIB.solution}, Best fitness Híbrido: {g_best_HIB.target.fitness}")

# # # --- LA PRUEBA DE FUEGO ---
# # print("¿Tiene diversidad?:", hasattr(model.history, 'list_diversity'))
# # print("¿Tiene exploración?:", hasattr(model.history, 'list_exploration'))

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
