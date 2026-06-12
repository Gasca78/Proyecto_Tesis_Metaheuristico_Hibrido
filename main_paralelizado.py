# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:14:52 2026

@author: oswal
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import concurrent.futures
from mealpy import FloatVar, DE
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
from benchmarks import modelos

# ==========================================
# 1. FUNCIÓN OBRERA (WORKER)
# Corre solo una vez (1 sola semilla)
# ==========================================
def corrida_individual(run_id, Modelo_Clase, config_modelo, function, epochs, pop_size):
    """
    Instancia el modelo, corre la optimización y extrae TODA la data.
    Retorna un diccionario con los resultados de esta corrida específica.
    """
    problem_dict = {
        "bounds": FloatVar(lb=function.lb, ub=function.ub), 
        "minmax": "min", 
        "obj_func": function.evaluate, 
        "name": function.name,
        "log_to": None
        }
    
    # Instanciamos el modelo con los parámetros que vengan en config_modelo
    model = Modelo_Clase(epoch=epochs, pop_size=pop_size, **config_modelo)
    
    # Correr y Medir
    start_time = time.time()
    g_best = model.solve(problem_dict)
    end_time = time.time()
    
    # Extraemos todos nuestros datos
    resultado = {
        'run_id': run_id,
        'fitness': g_best.target.fitness,
        'execution_time': end_time - start_time,
        'convergencia': model.history.list_global_best_fit[:epochs],
        'diversidad': model.history.list_diversity[:epochs],
        'exploracion': model.history.list_exploration[:epochs],
        'explotacion': model.history.list_exploitation[:epochs],
        'uso_DE': getattr(model, 'list_usage_DE', [0]*epochs)[:epochs],
        'uso_PSO': getattr(model, 'list_usage_PSO', [0]*epochs)[:epochs],
        'uso_GA': getattr(model, 'list_usage_GA', [0]*epochs)[:epochs],
    }
    
    return resultado

# Función para guardado en CSV
def guardar_csv(raw, nombre_archivo, name):
    df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw.items()]))
    df_temp.index.name = name
    df_temp.index += 1
    df_temp.to_csv(nombre_archivo)

# ========================================================
# 2. EL BLOQUE PRINCIPAL 
# ========================================================
if __name__=="__main__":

    # ==========================================
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ==========================================
    dims = config.DIMS
    runs = config.RUNS
    epochs = config.EPOCHS
    # epochs = config.EPOCHS_COMBINATORIA
    pop_size = config.POP_SIZE
    
    # ==========================================
    # CARGA DE FUNCIONES
    # ==========================================
    functions = benchmark_CEC2017.functions
    # functions = problemas_ingenieria.problems
    # functions = tsp.problems

    # ==========================================
    # CONFIGURACIÓN ESTUDIO DE SENSIBILIDAD
    # ==========================================
    modelos_a_probar = modelos.models
    
    for modelo_config in modelos_a_probar:
    
        # ==========================================
        # CONFIGURACIÓN PARA GUARDADO CSV
        # ==========================================
        Modelo_Clase = modelo_config['modelo']
        nombre_modelo = modelo_config['nombre_carpeta']
        parametros_modelo = modelo_config['parametros']
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        folder_name = f"Resultados_{nombre_modelo}_{timestamp}_{dims}_dims"
        # folder_name = f"Resultados_{nombre_modelo}_{timestamp}"
        data_path = os.path.join(config.SENSIBILIDAD_DIR, folder_name)
        
        # Crear la carpeta físicamente
        os.makedirs(data_path, exist_ok=True)
        print(f">>> Carpeta de resultados creada en:\n {data_path}")
        
        # Nombre para el archivo de salida
        nombre_archivo_fitness      = os.path.join(data_path, "Fitness.csv")
        nombre_archivo_tiempo       = os.path.join(data_path, "Tiempos.csv")
        nombre_archivo_convergencia = os.path.join(data_path, "Convergencia.csv")
        nombre_archivo_diversidad   = os.path.join(data_path, "Diversidad.csv")
        nombre_archivo_exploracion  = os.path.join(data_path, "Exploracion.csv")
        nombre_archivo_explotacion  = os.path.join(data_path, "Explotacion.csv")
        nombre_archivo_uso_modelos  = os.path.join(data_path, "Uso_Modelos.csv")
        
        # Diccionario para guardar TODOS los resultados crudos
        # Estructura: {'F1': [run1, run2...], 'F2': [run1, run2...]}
        raw_data = {} 
        raw_times = {}
        final_results = {}
        convergence_history = {}
        raw_diversity = {}
        raw_exploration = {}
        raw_exploitation = {}
        raw_usage_models = {}
        
        print(f"\n{'='*60}")
        print(f">>> EVALUANDO MODELO: {nombre_modelo}")
        print("="*60)
        
        for function in functions:
            print(f" -> Procesando función: {function.name}")
                        
            # Arreglos temporales para guardar los 30 resultados que irán llegando
            run_fitnesses = np.zeros(runs) # Lista temporal para los 30 fitness de ESTA función
            run_times = np.zeros(runs)
            fitness_per_epochs = np.zeros((runs, epochs))
            diversity_per_epochs = np.zeros((runs, epochs))
            exploration_per_epochs = np.zeros((runs, epochs))
            exploitation_per_epochs = np.zeros((runs, epochs))
            usage_DE_per_epochs = np.zeros((runs, epochs))
            usage_PSO_per_epochs = np.zeros((runs, epochs))
            usage_GA_per_epochs = np.zeros((runs, epochs))
            
            start_global = time.time()
            
            # --- INICIO DEL PROCESAMIENTO EN PARALELO ---
            # max_workers = Número de núcleos 
            with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
                # 1. Enviamos las 30 tareas al procesador
                futuros = []
                for i in range(runs):
                    tarea = executor.submit(corrida_individual, i, Modelo_Clase, parametros_modelo, function, epochs, pop_size)
                    futuros.append(tarea)
                # 2. Recibimos los resultados conforme va terminando
                for futuro in concurrent.futures.as_completed(futuros):
                    try:
                        data = futuro.result() # Obtenemos el diccionario resultado
                        idx = data['run_id']    
                        
                        run_fitnesses[idx] = data['fitness']
                        run_times[idx] = data['execution_time']
                        fitness_per_epochs[idx,:] = data['convergencia']
                        diversity_per_epochs[idx,:] = data['diversidad']
                        exploration_per_epochs[idx,:] = data['exploracion']
                        exploitation_per_epochs[idx,:] = data['explotacion']
                        usage_DE_per_epochs[idx,:] = data['uso_DE']
                        usage_PSO_per_epochs[idx,:] = data['uso_PSO']
                        usage_GA_per_epochs[idx,:] = data['uso_GA']
                        
                        print(f"[OK] Corrida {idx+1} terminada - Fit: {data['fitness']:.4E}")
                    except Exception as e:
                        print(f"[ERROR] Una corrida falló: {e}")
                        
            end_global = time.time()
            print(f"\n¡Todas las corridas terminaron! Tiempo total global: {(end_global-start_global)/60:.2f} minutos")
            
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
      
    # ==========================================
    # GUARDADO DE CSV
    # ==========================================
    print("\n" + "="*60)
    print(">>> EXPERIMENTO FINALIZADO")