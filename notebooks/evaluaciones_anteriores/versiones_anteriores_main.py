# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:36:33 2026

@author: oswal
"""

####################################################
# VERSIONES ANTERIORES PARA MAIN
####################################################

############## LO QUE SE TRABAJÓ CON BENCHMARK 2017 ##############

# # from mealpy import FloatVar, DE
# # from HIBRIDO import hibrid_JADE
# import opfunu
# # import numpy as np
# # import time
# # import pandas as pd 
# # import os
# # import datetime as dt
# import config


# # # ==========================================
# # # CONFIGURACIÓN DEL EXPERIMENTO
# # # ==========================================
# dims = config.DIMS
# # runs = 30 
# # epochs = 1000 
# # pop_size = 100

# # # Selección del Modelo
# # Modelo_Clase = hibrid_JADE
# # # Modelo_Clase = DE.JADE

# # # 1. Definir Directorio Base (Donde está este script)
# # try:
# #     base_path = os.path.dirname(os.path.abspath(__file__))
# # except NameError:
# #     base_path = os.getcwd() # Fallback por si corres en consola interactiva

# # # 2. Crear Nombre de Carpeta (Ej: "Resultados_hibrid_JADE_2025-12-16_14-30")
# # timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
# # folder_name = f"Resultados_{Modelo_Clase.__name__}_{timestamp}_{dims}_dims"
# # data_path = os.path.join(base_path, folder_name)

# # # 3. Crear la carpeta físicamente
# # os.makedirs(data_path, exist_ok=True)
# # print(f">>> Carpeta de resultados creada en:\n    {data_path}")
    
# # # Nombre para el archivo de salida
# # nombre_archivo_fitness      = os.path.join(data_path, "Fitness.csv")
# # nombre_archivo_tiempo       = os.path.join(data_path, "Tiempos.csv")
# # nombre_archivo_convergencia = os.path.join(data_path, "Convergencia.csv")
# # nombre_archivo_diversidad   = os.path.join(data_path, "Diversidad.csv")
# # nombre_archivo_exploracion  = os.path.join(data_path, "Exploracion.csv")
# # nombre_archivo_explotacion  = os.path.join(data_path, "Explotacion.csv")
# # # nombre_archivo_fitness = f"Resultados_fitness_{Modelo_Clase.__name__}_CEC2017.csv"
# # # nombre_archivo_tiempo = f"Resultados_tiempo_{Modelo_Clase.__name__}_CEC2017.csv"
# # # nombre_archivo_convergencia = f"Resultados_convergencia_{Modelo_Clase.__name__}_CEC2017.csv"
# # # nombre_archivo_diversidad = f"Resultados_diversidad_{Modelo_Clase.__name__}_CEC2017.csv"
# # # nombre_archivo_explotacion = f"Resultados_exploracion_{Modelo_Clase.__name__}_CEC2017.csv"
# # # nombre_archivo_exploracion = f"Resultados_explotacion_{Modelo_Clase.__name__}_CEC2017.csv"

# # # Función para guardado en CSV
# # def guardar_csv(raw, nombre_archivo, name):
# #     df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw.items() ]))
# #     df_temp.index.name = name
# #     df_temp.index += 1
# #     df_temp.to_csv(nombre_archivo)

# # ==========================================
# # CARGA DE FUNCIONES (CEC 2027)
# # ==========================================
# # Nota: CEC2017 tiene funciones F1 a F29
# functions = [
#     opfunu.cec_based.cec2017.F12017(ndim=dims),
#     opfunu.cec_based.cec2017.F22017(ndim=dims),
#     opfunu.cec_based.cec2017.F32017(ndim=dims),
#     opfunu.cec_based.cec2017.F42017(ndim=dims),
#     opfunu.cec_based.cec2017.F52017(ndim=dims),
#     opfunu.cec_based.cec2017.F62017(ndim=dims),
#     opfunu.cec_based.cec2017.F72017(ndim=dims),
#     opfunu.cec_based.cec2017.F82017(ndim=dims),
#     opfunu.cec_based.cec2017.F92017(ndim=dims),
#     opfunu.cec_based.cec2017.F102017(ndim=dims),
#     opfunu.cec_based.cec2017.F112017(ndim=dims),
#     opfunu.cec_based.cec2017.F122017(ndim=dims),
#     opfunu.cec_based.cec2017.F132017(ndim=dims),
#     opfunu.cec_based.cec2017.F142017(ndim=dims),
#     opfunu.cec_based.cec2017.F152017(ndim=dims),
#     opfunu.cec_based.cec2017.F162017(ndim=dims),
#     opfunu.cec_based.cec2017.F172017(ndim=dims),
#     opfunu.cec_based.cec2017.F182017(ndim=dims),
#     opfunu.cec_based.cec2017.F192017(ndim=dims),
#     opfunu.cec_based.cec2017.F202017(ndim=dims),
#     opfunu.cec_based.cec2017.F212017(ndim=dims),
#     opfunu.cec_based.cec2017.F222017(ndim=dims),
#     opfunu.cec_based.cec2017.F232017(ndim=dims),
#     opfunu.cec_based.cec2017.F242017(ndim=dims),
#     opfunu.cec_based.cec2017.F252017(ndim=dims),
#     opfunu.cec_based.cec2017.F262017(ndim=dims),
#     opfunu.cec_based.cec2017.F272017(ndim=dims),
#     opfunu.cec_based.cec2017.F282017(ndim=dims),
#     opfunu.cec_based.cec2017.F292017(ndim=dims)
# ]

# # # Funciones de Prueba para revisar resultados (resultados mostrados con probs fijas)
# # functions = [
# #     # opfunu.cec_based.cec2017.F12017(ndim=dims), # Empate
# #     # opfunu.cec_based.cec2017.F72017(ndim=dims), # No diferencias
# #     # opfunu.cec_based.cec2017.F82017(ndim=dims), # Perdedor
# #     # opfunu.cec_based.cec2017.F92017(ndim=dims), # Ganador
# #     # opfunu.cec_based.cec2017.F262017(ndim=dims), # Perdedor (compleja)
# #     # opfunu.cec_based.cec2017.F272017(ndim=dims), # Empate (compleja)
# #     # Seguir desde aquí
# #     opfunu.cec_based.cec2017.F282017(ndim=dims), # No diferencias
# #     opfunu.cec_based.cec2017.F292017(ndim=dims) # Ganador (compleja)
# # ]

# # # Diccionario para guardar TODOS los resultados crudos
# # # Estructura: {'F1': [run1, run2...], 'F2': [run1, run2...]}
# # raw_data = {} 
# # raw_times = {}
# # final_results = {}
# # convergence_history = {}
# # raw_diversity = {}
# # raw_exploration = {}
# # raw_exploitation = {}

# # print(f">>> INICIANDO EXPERIMENTO CON: {Modelo_Clase.__name__}")
# # print(f">>> Guardando en: {nombre_archivo_fitness}")
# # print("="*60)

# # for function in functions:
# #     run_fitnesses = [] # Lista temporal para los 30 fitness de ESTA función
# #     run_times = []
# #     fitness_per_epochs = np.zeros((runs, epochs))
# #     diversity_per_epochs = np.zeros((runs, epochs))
# #     exploration_per_epochs = np.zeros((runs, epochs))
# #     exploitation_per_epochs = np.zeros((runs, epochs))
    
# #     print(f"\nProcesando: {function.name} ...")
    
# #     for i in range(runs):
# #         # 1. Instanciar modelo limpio
# #         model = Modelo_Clase(epoch=epochs, pop_size=pop_size)

# #         # 2. Configurar problema
# #         problem_dict = {
# #             "bounds": FloatVar(lb=function.lb, ub=function.ub),
# #             "minmax": "min",
# #             "obj_func": function.evaluate,
# #             "log_to": None
# #         }
        
# #         # 3. Correr y Medir
# #         start_time = time.time()
# #         g_best = model.solve(problem_dict)
# #         end_time = time.time()
        
# #         fitness = g_best.target.fitness
# #         execution_time = end_time - start_time
        
# #         # Guardado por época, en cada corrida qué paso en cada época
# #         fitness_per_epochs[i, :] = model.history.list_global_best_fit[:epochs]
# #         diversity_per_epochs[i, :] = model.history.list_diversity[:epochs]
# #         exploration_per_epochs[i, :] = model.history.list_exploration[:epochs]
# #         exploitation_per_epochs[i, :] = model.history.list_exploitation[:epochs]
        
# #         # 4. Guardar datos
# #         run_fitnesses.append(fitness)
# #         run_times.append(execution_time)
        
# #         # Feedback visual minimalista (para no saturar la consola con 300 líneas)
# #         # Imprime un punto por cada run, y el fitness al final de la línea cada 5 o 10
# #         if (i+1) % 5 == 0:
# #              print(f"  Run {i+1}/{runs} -> Fit: {fitness:.6E} | Tiempo: {execution_time:.2f} s")

# #     # --- AL TERMINAR LAS 30 CORRIDAS DE LA FUNCIÓN ---

# #     # 1. Guardar en el diccionario maestro (Esto es lo que irá al CSV)
# #     # Para los fitness y tiempo promedio por corrida
# #     raw_data[function.name] = run_fitnesses
# #     raw_times[function.name] = run_times
# #     # Para la convergencia, diversidad, exploracion y explotación por época (Trayectorias promedio)
# #     convergence_history[function.name] = np.mean(fitness_per_epochs, axis=0)
# #     raw_diversity[function.name] = np.mean(diversity_per_epochs, axis=0)
# #     raw_exploration[function.name] = np.mean(exploration_per_epochs, axis=0)
# #     raw_exploitation[function.name] = np.mean(exploitation_per_epochs, axis=0)
    
# #     # 2. GUARDADO DE SEGURIDAD (Progressive Save)
# #     # Esto sobrescribe el archivo cada vez que termina una función.
# #     try:
# #         # Para el fitness
# #         guardar_csv(raw_data, nombre_archivo_fitness, 'Run_ID')
# #         # Para el tiempo
# #         guardar_csv(raw_times, nombre_archivo_tiempo, 'Run_ID')
# #         # Para la convergencia
# #         guardar_csv(convergence_history, nombre_archivo_convergencia, 'Epoca')
# #         # Para la diversidad
# #         guardar_csv(raw_diversity, nombre_archivo_diversidad, 'Epoca')
# #         # Para la exploracion
# #         guardar_csv(raw_exploration, nombre_archivo_exploracion, 'Epoca')
# #         # Para la explotacion
# #         guardar_csv(raw_exploitation, nombre_archivo_explotacion, 'Epoca')
# #     except Exception as e:
# #         print(f"⚠️ Advertencia: No se pudo guardar el temporal ({e})")
# #     print("  >>> Guardado parcial exitoso.")
    
# #     # Cálculos estadísticos
# #     mean_fit = np.mean(run_fitnesses)
# #     std_fit = np.std(run_fitnesses)
# #     mean_time = np.mean(run_times)
    
# #     # Guardar para el reporte final (usando el nombre como clave)
# #     final_results[function.name] = {
# #         'mean': mean_fit,
# #         'best': np.min(run_fitnesses),
# #         'worst': np.max(run_fitnesses),
# #         'std': std_fit
# #     }

# #     # Reporte Individual
# #     print("-" * 50)
# #     print(f"RESUMEN: {function.name}")
# #     print(f"  Mejor : {np.min(run_fitnesses):.6E}")
# #     print(f"  Peor  : {np.max(run_fitnesses):.6E}")
# #     print(f"  Media : {mean_fit:.6E}")
# #     print(f"  Std   : {std_fit:.6E}")
# #     print(f"  Tiempo Promedio: {mean_time:.2f} s")
# #     print("-" * 50)
# #     # # Reporte rápido en consola
# #     # print(f"  Resultados {function.name}: Mean={np.mean(run_fitnesses):.4E} | Std={np.std(run_fitnesses):.4E}")

# # # ==========================================
# # # GUARDADO DE CSV
# # # ==========================================
# # print("\n" + "="*60)
# # print("EXPERIMENTO FINALIZADO")

# # ==========================================
# # Opción para mostrar el dataframe creado
# # ==========================================

# # # Volvemos a generar el DF solo para mostrarlo en pantalla, 
# # # el archivo ya debería estar guardado por el bucle anterior.
# # df_final = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw_data.items() ]))
# # df_final.index.name = 'Run_ID'
# # df_final.index += 1
# # df_final_t = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw_times.items() ]))
# # df_final_t.index.name = 'Run_ID'
# # df_final_t.index += 1

# # # Opcional: Forzar un último guardado "por si acaso"
# # df_final.to_csv(nombre_archivo_fitness)
# # df_final_t.to_csv(nombre_archivo_tiempo)

# # print(f"Archivo final verificado: {nombre_archivo_fitness}")
# # print("Muestra de los datos:")
# # print(df_final.head())
# # print("="*60)

############## LO QUE SE TRABAJÓ CON BENCHMARK 2020 ##############

# # from mealpy import FloatVar, DE
# # from HIBRIDO import hibrid_JADE
# import opfunu
# # import numpy as np
# # import time
# # import pandas as pd  
# import config

# # # ==========================================
# # # CONFIGURACIÓN DEL EXPERIMENTO
# # # ==========================================
# dims = config.DIMS
# # runs = 30 
# # epochs = 1000 
# # pop_size = 100

# # # Elige el modelo a probar (Descomenta el que vayas a usar)
# # # Esto define también el nombre del archivo CSV
# # # Modelo_Clase = hibrid_JADE
# # # Modelo_Clase = DE.JADE 

# # # Nombre para el archivo de salida
# # nombre_archivo = f"Resultados_{Modelo_Clase.__name__}_CEC2020.csv"

# # ==========================================
# # CARGA DE FUNCIONES (CEC 2020)
# # ==========================================
# # Nota: CEC2020 tiene funciones F1 a F10
# functions = [
#     opfunu.cec_based.cec2020.F12020(ndim=dims),
#     opfunu.cec_based.cec2020.F22020(ndim=dims),
#     opfunu.cec_based.cec2020.F32020(ndim=dims),
#     opfunu.cec_based.cec2020.F42020(ndim=dims),
#     opfunu.cec_based.cec2020.F52020(ndim=dims),
#     opfunu.cec_based.cec2020.F62020(ndim=dims),
#     opfunu.cec_based.cec2020.F72020(ndim=dims),
#     opfunu.cec_based.cec2020.F82020(ndim=dims),
#     opfunu.cec_based.cec2020.F92020(ndim=dims),
#     opfunu.cec_based.cec2020.F102020(ndim=dims),
# ]

# # # Diccionario para guardar TODOS los resultados crudos
# # # Estructura: {'F1': [run1, run2...], 'F2': [run1, run2...]}
# # raw_data = {} 
# # final_results = {}

# # print(f">>> INICIANDO EXPERIMENTO CON: {Modelo_Clase.__name__}")
# # print(f">>> Guardando en: {nombre_archivo}")
# # print("="*60)

# # for function in functions:
# #     run_fitnesses = [] # Lista temporal para los 30 fitness de ESTA función
# #     run_times = []
    
# #     print(f"\nProcesando: {function.name} ...")
    
# #     for i in range(runs):
# #         # 1. Instanciar modelo limpio
# #         model = Modelo_Clase(epoch=epochs, pop_size=pop_size)

# #         # 2. Configurar problema
# #         problem_dict = {
# #             "bounds": FloatVar(lb=function.lb, ub=function.ub),
# #             "minmax": "min",
# #             "obj_func": function.evaluate,
# #             "log_to": None
# #         }
        
# #         # 3. Correr y Medir
# #         start_time = time.time()
# #         g_best = model.solve(problem_dict)
# #         end_time = time.time()
        
# #         fitness = g_best.target.fitness
# #         execution_time = end_time - start_time
        
# #         # 4. Guardar datos
# #         run_fitnesses.append(fitness)
# #         run_times.append(execution_time)
        
# #         # Feedback visual minimalista (para no saturar la consola con 300 líneas)
# #         # Imprime un punto por cada run, y el fitness al final de la línea cada 5 o 10
# #         if (i+1) % 5 == 0:
# #              print(f"  Run {i+1}/{runs} -> Fit: {fitness:.6E} | Tiempo: {execution_time:.2f} s")

# #     # --- AL TERMINAR LAS 30 CORRIDAS DE LA FUNCIÓN ---
    
# #     # 1. Guardar en el diccionario maestro (Esto es lo que irá al CSV)
# #     raw_data[function.name] = run_fitnesses
    
# #     # 2. GUARDADO DE SEGURIDAD (Progressive Save)
# #     # Esto sobrescribe el archivo cada vez que termina una función.
# #     try:
# #         df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw_data.items() ]))
# #         df_temp.index.name = 'Run_ID'
# #         df_temp.index += 1
# #         df_temp.to_csv(nombre_archivo)
# #     except Exception as e:
# #         print(f"⚠️ Advertencia: No se pudo guardar el temporal ({e})")
# #     print("  >>> Guardado parcial exitoso.")
    
# #     # Cálculos estadísticos
# #     mean_fit = np.mean(run_fitnesses)
# #     std_fit = np.std(run_fitnesses)
# #     mean_time = np.mean(run_times)
    
# #     # Guardar para el reporte final (usando el nombre como clave)
# #     final_results[function.name] = {
# #         'mean': mean_fit,
# #         'best': np.min(run_fitnesses),
# #         'worst': np.max(run_fitnesses),
# #         'std': std_fit
# #     }

# #     # Reporte Individual
# #     print("-" * 50)
# #     print(f"RESUMEN: {function.name}")
# #     print(f"  Mejor : {np.min(run_fitnesses):.6E}")
# #     print(f"  Peor  : {np.max(run_fitnesses):.6E}")
# #     print(f"  Media : {mean_fit:.6E}")
# #     print(f"  Std   : {std_fit:.6E}")
# #     print(f"  Tiempo Promedio: {mean_time:.2f} s")
# #     print("-" * 50)
# #     # Reporte rápido en consola
# #     print(f"  Resultados {function.name}: Mean={np.mean(run_fitnesses):.4E} | Std={np.std(run_fitnesses):.4E}")

# # # ==========================================
# # # GUARDADO DE CSV
# # # ==========================================
# # print("\n" + "="*60)
# # print("EXPERIMENTO FINALIZADO")

# # # Volvemos a generar el DF solo para mostrarlo en pantalla, 
# # # el archivo ya debería estar guardado por el bucle anterior.
# # df_final = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw_data.items() ]))
# # df_final.index.name = 'Run_ID'
# # df_final.index += 1

# # # Opcional: Forzar un último guardado "por si acaso"
# # df_final.to_csv(nombre_archivo)

# # print(f"Archivo final verificado: {nombre_archivo}")
# # print("Muestra de los datos:")
# # print(df_final.head())
# # print("="*60)
# #-------------------------------------------------------------------------------------------------------
# # from mealpy import FloatVar
# # from mealpy import DE
# # from HIBRIDO import hibrid_JADE
# # import opfunu
# # import numpy as np
# # import pandas as pd
# # import time

# # # Configuración Global
# # dims = 30
# # runs = 3
# # epochs = 1000
# # pop_size = 50

# # # Lista de Funciones CEC 2022
# # functions = [
# #     opfunu.cec_based.cec2020.F12020(ndim=dims),
# #     opfunu.cec_based.cec2020.F22020(ndim=dims),
# #     opfunu.cec_based.cec2020.F32020(ndim=dims),
# #     opfunu.cec_based.cec2020.F42020(ndim=dims),
# #     opfunu.cec_based.cec2020.F52020(ndim=dims),
# #     opfunu.cec_based.cec2020.F62020(ndim=dims),
# #     opfunu.cec_based.cec2020.F72020(ndim=dims),
# #     opfunu.cec_based.cec2020.F82020(ndim=dims),
# #     opfunu.cec_based.cec2020.F92020(ndim=dims),
# #     opfunu.cec_based.cec2020.F102020(ndim=dims)
# #     # opfunu.cec_based.cec2020.F112020(ndim=dims),
# #     # opfunu.cec_based.cec2020.F122020(ndim=dims)
# # ]

# # # Diccionario para guardar el resumen final por función
# # # Estructura: {'F1': {'mean': 0.5, 'std': 0.1}, 'F2': ...}
# # final_results = {} 

# # for function in functions:
# #     # --- CORRECCIÓN CRÍTICA: Reiniciar listas por CADA función ---
# #     hist_fitness = []
# #     hist_times = []
    
# #     print(f"\n>>> Iniciando Benchmark para: {function.name}")
    
# #     for i in range(runs):
# #         # Instanciar modelo nuevo en cada run para limpiar memoria
# #         model = hibrid_JADE(epoch=epochs, pop_size=pop_size)
# #         # model = DE.JADE(epoch=epochs, pop_size=pop_size)
        
# #         problem_dict = {
# #             "bounds": FloatVar(lb=function.lb, ub=function.ub),
# #             "minmax": "min",
# #             "obj_func": function.evaluate,
# #             "log_to": None
# #         }
        
# #         # Ejecución y medición
# #         start_time = time.time()
# #         g_best = model.solve(problem_dict)
# #         end_time = time.time()
        
# #         execution_time = end_time - start_time
# #         fitness = g_best.target.fitness
        
# #         hist_fitness.append(fitness)
# #         hist_times.append(execution_time)
        
# #         print(f"  Run {i+1}/{runs} -> Fitness: {fitness:.6E} | Tiempo: {execution_time:.2f} s")
    
# #     # Cálculos estadísticos
# #     mean_fit = np.mean(hist_fitness)
# #     std_fit = np.std(hist_fitness)
# #     mean_time = np.mean(hist_times)
    
# #     # Guardar para el reporte final (usando el nombre como clave)
# #     final_results[function.name] = {
# #         'mean': mean_fit,
# #         'best': np.min(hist_fitness),
# #         'worst': np.max(hist_fitness),
# #         'std': std_fit
# #     }

# #     # Reporte Individual
# #     print("-" * 50)
# #     print(f"RESUMEN: {function.name}")
# #     print(f"  Mejor : {np.min(hist_fitness):.6E}")
# #     print(f"  Peor  : {np.max(hist_fitness):.6E}")
# #     print(f"  Media : {mean_fit:.6E}")
# #     print(f"  Std   : {std_fit:.6E}")
# #     print(f"  Tiempo Promedio: {mean_time:.2f} s")
# #     print("-" * 50)

# # # Reporte Global Final
# # print("\n" + "="*60)
# # print("             TABLA FINAL DE RESULTADOS (CEC 2022)")
# # print("="*60)
# # print(f"{'Función':<15} | {'Media Fitness':<20} | {'Mejor Fitness':<20}")
# # print("-" * 60)

# # best_func_name = ""
# # best_val = float('inf')

# # for func_name, stats in final_results.items():
# #     print(f"{func_name:<15} | {stats['mean']:<20.6E} | {stats['best']:<20.6E}")
    
# #     # Buscar cuál fue la función donde mejor le fue (menor media)
# #     if stats['mean'] < best_val:
# #         best_val = stats['mean']
# #         best_func_name = func_name

# # print("="*60)
# # print(f"Mejor Desempeño General: {best_func_name} con media {best_val:.6E}")