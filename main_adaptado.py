# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:14:52 2026

@author: oswal
"""

import argparse # <--- LIBRERÍA NUEVA PARA LEER ARGUMENTOS
from mealpy import FloatVar, DE
import HIBRIDO_sin_filtro
import HIBRIDO
import HIBRIDO_Markov_Estricto
import HIBRIDO_pensante
import HIBRIDO_pensante_sin_filtro
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
# LECTURA DE ARGUMENTOS DESDE TERMINAL
# ==========================================
parser = argparse.ArgumentParser(description="Ejecutor de Metaheurísticas")
parser.add_argument("--modelo", type=str, required=True, help="Nombre del modelo a ejecutar")
parser.add_argument("--benchmark", type=str, required=True, help="Benchmark a resolver (cec2017, tsp, ingenieria)")
args = parser.parse_args()

# ==========================================
# MAPEO DE MODELOS Y BENCHMARKS
# ==========================================
# Diccionario para seleccionar la clase dinámicamente
diccionario_modelos = {
    "hibrid_JADE": HIBRIDO.hibrid_JADE,
    "Markov_WTA": HIBRIDO_Markov_Estricto.hibrid_JADE_Markov_WTA,
    "Markov_80_15_5": HIBRIDO.hibrid_JADE, # 80-15-5
    "probs_sin_filtro": HIBRIDO_pensante_sin_filtro.hibrid_JADE_probs_sin_filtro,
    "JADE": DE.JADE
}

# Diccionario para seleccionar los problemas
diccionario_benchmarks = {
    "cec2017": benchmark_CEC2017.functions,
    "ingenieria": problemas_ingenieria.problems,
    "tsp": tsp.problems
}

# Asignación basada en lo que recibe la terminal
Modelo_Clase = diccionario_modelos[args.modelo]
functions = diccionario_benchmarks[args.benchmark]

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
dims = config.DIMS
runs = config.RUNS
epochs = config.EPOCHS
# epochs = config.EPOCHS_COMBINATORIA # (Descomentar en tu lógica interna si es necesario)
pop_size = config.POP_SIZE

# 2. Crear Nombre de Carpeta (Añadiendo el nombre del benchmark para no sobreescribir)
timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
if functions == "cec2017":
    folder_name = f"Resultados_{Modelo_Clase.__name__}_{args.benchmark}_{timestamp}_{dims}_dims"
else:
    folder_name = f"Resultados_{Modelo_Clase.__name__}_{args.benchmark}_{timestamp}"
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
nombre_archivo_uso_modelos  = os.path.join(data_path, "Uso_Modelos.csv")

def guardar_csv(raw, nombre_archivo, name):
    df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw.items() ]))
    df_temp.index.name = name
    df_temp.index += 1
    df_temp.to_csv(nombre_archivo)
    
# Variables de almacenamiento (sin cambios)
raw_data = {} 
raw_times = {}
final_results = {}
convergence_history = {}
raw_diversity = {}
raw_exploration = {}
raw_exploitation = {}
raw_usage_models = {}

print(f">>> INICIANDO EXPERIMENTO CON: {Modelo_Clase.__name__} en {args.benchmark.upper()}")
print(f">>> Guardando en: {nombre_archivo_fitness}")
print("="*60)

for function in functions:
    run_fitnesses = [] # Lista temporal para los 30 fitness de ESTA función
    run_times = []
    fitness_per_epochs = np.zeros((runs, epochs))
    diversity_per_epochs = np.zeros((runs, epochs))
    exploration_per_epochs = np.zeros((runs, epochs))
    exploitation_per_epochs = np.zeros((runs, epochs))
    usage_DE_per_epochs = np.zeros((runs, epochs))
    usage_PSO_per_epochs = np.zeros((runs, epochs))
    usage_GA_per_epochs = np.zeros((runs, epochs))
    
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
        usage_DE_per_epochs[i, :] = model.list_usage_DE[:epochs]
        usage_PSO_per_epochs[i, :] = model.list_usage_PSO[:epochs]
        usage_GA_per_epochs[i, :] = model.list_usage_GA[:epochs]
        
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
    raw_usage_models[f"{function.name}_DE"] = np.mean(usage_DE_per_epochs, axis=0)
    raw_usage_models[f"{function.name}_PSO"] = np.mean(usage_PSO_per_epochs, axis=0)
    raw_usage_models[f"{function.name}_GA"] = np.mean(usage_GA_per_epochs, axis=0)
    
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
        # Para el uso de los modelos
        guardar_csv(raw_usage_models, nombre_archivo_uso_modelos, 'Epoca')
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