# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:14:52 2026

@author: oswal
"""

from mealpy import FloatVar, DE
import HIBRIDO
import HIBRIDO_Markov_Estricto
import HIBRIDO_pensante
import opfunu
import numpy as np
import time
import pandas as pd 
import os
import datetime as dt
import config 
from benchmarks import benchmark_CEC2017
from benchmarks import RWCO_2020

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
# dims = config.DIMS
runs = config.RUNS
epochs = config.EPOCHS
pop_size = config.POP_SIZE

# Selección del Modelo
# Modelo_Clase = HIBRIDO.hibrid_JADE # Markov con Inercia
# Modelo_Clase = HIBRIDO_Markov_Estricto.hibrid_JADE # Markov WTA
# Modelo_Clase = HIBRIDO_pensante.hibrid_JADE # Sin Markov, solo probabilidades cambiantes
# Modelo_Clase = DE.JADE

# 2. Crear Nombre de Carpeta (Ej: "Resultados_hibrid_JADE_2025-12-16_14-30")
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
# folder_name = f"Resultados_{Modelo_Clase.__name__}_{timestamp}_{dims}_dims"
folder_name = f"Resultados_{Modelo_Clase.__name__}_{timestamp}"
data_path = os.path.join(config.RESULTS_DIR, folder_name)

# 3. Crear la carpeta físicamente
os.makedirs(data_path, exist_ok=True)
print(f">>> Carpeta de resultados creada en:\n    {data_path}")

# Nombre para el archivo de salida
nombre_archivo_fitness      = os.path.join(data_path, "Fitness.csv")
nombre_archivo_tiempo       = os.path.join(data_path, "Tiempos.csv")
nombre_archivo_convergencia = os.path.join(data_path, "Convergencia.csv")
nombre_archivo_diversidad   = os.path.join(data_path, "Diversidad.csv")
nombre_archivo_exploracion  = os.path.join(data_path, "Exploracion.csv")
nombre_archivo_explotacion  = os.path.join(data_path, "Explotacion.csv")

# Función para guardado en CSV
def guardar_csv(raw, nombre_archivo, name):
    df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw.items() ]))
    df_temp.index.name = name
    df_temp.index += 1
    df_temp.to_csv(nombre_archivo)
    
# ==========================================
# CARGA DE FUNCIONES
# ==========================================
# functions = benchmark_CEC2017.functions
functions = RWCO_2020.problems

# Diccionario para guardar TODOS los resultados crudos
# Estructura: {'F1': [run1, run2...], 'F2': [run1, run2...]}
raw_data = {} 
raw_times = {}
final_results = {}
convergence_history = {}
raw_diversity = {}
raw_exploration = {}
raw_exploitation = {}

print(f">>> INICIANDO EXPERIMENTO CON: {Modelo_Clase.__name__}")
print(f">>> Guardando en: {nombre_archivo_fitness}")
print("="*60)

for function in functions:
    run_fitnesses = [] # Lista temporal para los 30 fitness de ESTA función
    run_times = []
    fitness_per_epochs = np.zeros((runs, epochs))
    diversity_per_epochs = np.zeros((runs, epochs))
    exploration_per_epochs = np.zeros((runs, epochs))
    exploitation_per_epochs = np.zeros((runs, epochs))
    
    print(f"\nProcesando: {function.name} ...")
    
    for i in range(runs):
        # 1. Instanciar modelo limpio
        model = Modelo_Clase(epoch=epochs, pop_size=pop_size)

        # 2. Configurar problema
        problem_dict = {
            "bounds": FloatVar(lb=function.lb, ub=function.ub),
            "minmax": "min",
            "obj_func": function.evaluate,
            "log_to": None
        }
        
        # 3. Correr y Medir
        start_time = time.time()
        g_best = model.solve(problem_dict)
        end_time = time.time()
        
        fitness = g_best.target.fitness
        execution_time = end_time - start_time
        
        # Guardado por época, en cada corrida qué paso en cada época
        fitness_per_epochs[i, :] = model.history.list_global_best_fit[:epochs]
        diversity_per_epochs[i, :] = model.history.list_diversity[:epochs]
        exploration_per_epochs[i, :] = model.history.list_exploration[:epochs]
        exploitation_per_epochs[i, :] = model.history.list_exploitation[:epochs]
        
        # 4. Guardar datos
        run_fitnesses.append(fitness)
        run_times.append(execution_time)
        
        # Feedback visual minimalista (para no saturar la consola con 300 líneas)
        # Imprime un punto por cada run, y el fitness al final de la línea cada 5 o 10
        if (i+1) % 5 == 0:
             print(f"  Run {i+1}/{runs} -> Fit: {fitness:.6E} | Tiempo: {execution_time:.2f} s")

    # --- AL TERMINAR LAS 30 CORRIDAS DE LA FUNCIÓN ---

    # 1. Guardar en el diccionario maestro (Esto es lo que irá al CSV)
    # Para los fitness y tiempo promedio por corrida
    raw_data[function.name] = run_fitnesses
    raw_times[function.name] = run_times
    # Para la convergencia, diversidad, exploracion y explotación por época (Trayectorias promedio)
    convergence_history[function.name] = np.mean(fitness_per_epochs, axis=0)
    raw_diversity[function.name] = np.mean(diversity_per_epochs, axis=0)
    raw_exploration[function.name] = np.mean(exploration_per_epochs, axis=0)
    raw_exploitation[function.name] = np.mean(exploitation_per_epochs, axis=0)
    
    # 2. GUARDADO DE SEGURIDAD (Progressive Save)
    # Esto sobrescribe el archivo cada vez que termina una función.
    try:
        # Para el fitness
        guardar_csv(raw_data, nombre_archivo_fitness, 'Run_ID')
        # Para el tiempo
        guardar_csv(raw_times, nombre_archivo_tiempo, 'Run_ID')
        # Para la convergencia
        guardar_csv(convergence_history, nombre_archivo_convergencia, 'Epoca')
        # Para la diversidad
        guardar_csv(raw_diversity, nombre_archivo_diversidad, 'Epoca')
        # Para la exploracion
        guardar_csv(raw_exploration, nombre_archivo_exploracion, 'Epoca')
        # Para la explotacion
        guardar_csv(raw_exploitation, nombre_archivo_explotacion, 'Epoca')
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo guardar el temporal ({e})")
    print("  >>> Guardado parcial exitoso.")
    
    # Cálculos estadísticos
    mean_fit = np.mean(run_fitnesses)
    std_fit = np.std(run_fitnesses)
    mean_time = np.mean(run_times)
    
    # Guardar para el reporte final (usando el nombre como clave)
    final_results[function.name] = {
        'mean': mean_fit,
        'best': np.min(run_fitnesses),
        'worst': np.max(run_fitnesses),
        'std': std_fit
    }

    # Reporte Individual
    print("-" * 50)
    print(f"RESUMEN: {function.name}")
    print(f"  Mejor : {np.min(run_fitnesses):.6E}")
    print(f"  Peor  : {np.max(run_fitnesses):.6E}")
    print(f"  Media : {mean_fit:.6E}")
    print(f"  Std   : {std_fit:.6E}")
    print(f"  Tiempo Promedio: {mean_time:.2f} s")
    print("-" * 50)
    # # Reporte rápido en consola
    # print(f"  Resultados {function.name}: Mean={np.mean(run_fitnesses):.4E} | Std={np.std(run_fitnesses):.4E}")

# ==========================================
# GUARDADO DE CSV
# ==========================================
print("\n" + "="*60)
print("EXPERIMENTO FINALIZADO")