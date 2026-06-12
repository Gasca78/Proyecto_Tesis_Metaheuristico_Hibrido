# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:59:20 2026

@author: oswal
"""
from mealpy import FloatVar, DE
from mealpy import Multitask
import HIBRIDO
import opfunu
import numpy as np
import time
import pandas as pd 
import os
import datetime as dt
import config 
from benchmarks import benchmark_CEC2017
from benchmarks import problemas_ingenieria
from benchmarks.tsp import tsp

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
dims = config.DIMS
runs = config.RUNS
epochs = config.EPOCHS
pop_size = config.POP_SIZE

# Definimos las funciones
f01 = opfunu.cec_based.cec2017.F12017(ndim=dims) # Empate
f02 = opfunu.cec_based.cec2017.F72017(ndim=dims) # No diferencias
f03 = opfunu.cec_based.cec2017.F82017(ndim=dims) # Perdedor
f04 = opfunu.cec_based.cec2017.F92017(ndim=dims) # Ganador
f05 = opfunu.cec_based.cec2017.F262017(ndim=dims) # Perdedor (compleja)
f06 = opfunu.cec_based.cec2017.F272017(ndim=dims) # Empate (compleja)
f07 = opfunu.cec_based.cec2017.F282017(ndim=dims) # No diferencias
f08 = opfunu.cec_based.cec2017.F292017(ndim=dims) # Ganador (compleja)

# Definimos los problemas
p01 = {"bounds": FloatVar(lb=f01.lb, ub=f01.ub), "minmax": "min", "obj_func": f01.evaluate, "name": f01.name,"log_to": None}
p02 = {"bounds": FloatVar(lb=f02.lb, ub=f01.ub), "minmax": "min", "obj_func": f02.evaluate, "name": f02.name,"log_to": None}
p03 = {"bounds": FloatVar(lb=f03.lb, ub=f01.ub), "minmax": "min", "obj_func": f03.evaluate, "name": f03.name,"log_to": None}
p04 = {"bounds": FloatVar(lb=f04.lb, ub=f01.ub), "minmax": "min", "obj_func": f04.evaluate, "name": f04.name,"log_to": None}
p05 = {"bounds": FloatVar(lb=f05.lb, ub=f01.ub), "minmax": "min", "obj_func": f05.evaluate, "name": f05.name,"log_to": None}
p06 = {"bounds": FloatVar(lb=f06.lb, ub=f01.ub), "minmax": "min", "obj_func": f06.evaluate, "name": f06.name,"log_to": None}
p07 = {"bounds": FloatVar(lb=f07.lb, ub=f01.ub), "minmax": "min", "obj_func": f07.evaluate, "name": f07.name,"log_to": None}
p08 = {"bounds": FloatVar(lb=f08.lb, ub=f01.ub), "minmax": "min", "obj_func": f08.evaluate, "name": f08.name,"log_to": None}

# Definimos los modelos
model01 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, matriz_type="moderado", success_filter=True, memory_type="markov")
model02 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, matriz_type="moderado", success_filter=True, memory_type="markov")
model03 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, matriz_type="moderado", success_filter=True, memory_type="markov")
model04 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, matriz_type="estricto", success_filter=True, memory_type="markov")
model05 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, matriz_type="estricto", success_filter=True, memory_type="markov")
model06 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, matriz_type="estricto", success_filter=True, memory_type="markov")
model07 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, matriz_type="conservador", success_filter=True, memory_type="markov")
model08 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, matriz_type="conservador", success_filter=True, memory_type="markov")
model09 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, matriz_type="conservador", success_filter=True, memory_type="markov")
model10 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, matriz_type="moderado", success_filter=False, memory_type="markov")
model11 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, matriz_type="moderado", success_filter=False, memory_type="markov")
model12 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, matriz_type="moderado", success_filter=False, memory_type="markov")
model13 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, matriz_type="estricto", success_filter=False, memory_type="markov")
model14 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, matriz_type="estricto", success_filter=False, memory_type="markov")
model15 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, matriz_type="estricto", success_filter=False, memory_type="markov")
model16 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, matriz_type="conservador", success_filter=False, memory_type="markov")
model17 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, matriz_type="conservador", success_filter=False, memory_type="markov")
model18 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, matriz_type="conservador", success_filter=False, memory_type="markov")
model19 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, success_filter=True, memory_type="probs")
model20 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, success_filter=True, memory_type="probs")
model21 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, success_filter=True, memory_type="probs")
model22 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=5, success_filter=False, memory_type="probs")
model23 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=10, success_filter=False, memory_type="probs")
model24 = HIBRIDO.hibrid_JADE(epoch=epochs, pop_size=pop_size, update_interval=50, success_filter=False, memory_type="probs")

problems = (p01, p02, p03, p04, p05, p06, p07, p08)
algorithms = (model01, model02, model03, model04, model05, model06, 
              model07, model08, model09, model10, model11, model12,
              model13, model14, model15, model16, model17, model18,
              model19, model20, model21, model22, model23, model24)

if __name__=="__main__":
    multitask = Multitask(algorithms=algorithms, problems=problems, n_workers=12)
    multitask.execute(n_trials=runs, save_path="C:/Users/oswal/OneDrive/Documentos/UdeG/Maestría/Tesis/Codigo/prueba_multitask", save_as="csv", save_convergence=True, verbose=False)