# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:14:52 2026

@author: oswal
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import concurrent.futures
from mealpy import FloatVar
import numpy as np
import time
import pandas as pd 
import os
import datetime as dt
import config 
import HIBRIDO
from benchmarks import benchmark_CEC2017
from benchmarks import problemas_ingenieria
from benchmarks.tsp import tsp
from benchmarks import modelos
from Aplicaciones_CD.bioinformatics import BioinformaticsFS, leukemia_enviroment
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# 1. FUNCIÓN OBRERA (WORKER)
# Corre solo una vez (1 sola semilla)
# ==========================================
def corrida_individual(run_id, Modelo_Clase, config_modelo, function, epochs, pop_size):
    """
    Instancia el modelo, corre la optimización y extrae TODA la data.
    Retorna un diccionario con los resultados de esta corrida específica.
    """
    if isinstance(function, dict) and function.get('is_tsp'):
        # Es un problema TSP: Construimos la instancia ADENTRO del trabajador
        from benchmarks.tsp.tsp import TSPInstance # Importación local
        tsp_inst = TSPInstance(function['filepath'])
        problem_dict = {
            "bounds": FloatVar(lb=tsp_inst.lb, ub=tsp_inst.ub), 
            "minmax": "min", 
            "obj_func": tsp_inst.evaluate, 
            "name": tsp_inst.name,
            "log_to": None
        }
    else:
        # Es un problema CEC2017 o Ingeniería normal
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
        'solution': g_best.solution,
        'execution_time': end_time - start_time,
        'convergencia': model.history.list_global_best_fit[:epochs],
        'diversidad': model.history.list_diversity[:epochs],
        'exploracion': model.history.list_exploration[:epochs],
        'explotacion': model.history.list_exploitation[:epochs],
        'uso_DE': getattr(model, 'list_usage_DE', [0]*epochs)[:epochs],
        'uso_PSO': getattr(model, 'list_usage_PSO', [0]*epochs)[:epochs],
        'uso_GA': getattr(model, 'list_usage_GA', [0]*epochs)[:epochs],
    }
    
    if hasattr(function, 'get_metrics'):
        metricas_extra = function.get_metrics(g_best.solution)
        resultado.update(metricas_extra)
    
    return resultado

# Función para guardado en CSV
def guardar_csv(raw, nombre_archivo, name):
    df_temp = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in raw.items()]))
    df_temp.index.name = name
    df_temp.index += 1
    df_temp.to_csv(nombre_archivo)
    
# Función para obtener valores finales después del feature selection
def get_final_values(pesos_finales, X_train, X_test, y_train, y_test, best_k, selected_gens):
    mascara_final = np.array(pesos_finales) > 0.5
    genes_seleccionados = selected_gens[mascara_final]
    X_train_final = X_train[:, mascara_final]
    X_test_final = X_test[:, mascara_final]
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_final, y_train)
    acc_final = knn.score(X_test_final, y_test)
    return acc_final, genes_seleccionados

# ========================================================
# 2. EL BLOQUE PRINCIPAL 
# ========================================================
if __name__=="__main__":

    # ==========================================
    # CONFIGURACIÓN DEL EXPERIMENTO
    # ==========================================
    dims = config.DIMS
    runs = config.RUNS
    # ---- COMENTA Y DESCOMENTA EN FUNCIÓN DE LOS PROBLEMAS (MATEMÁTICOS E INGENIERIA O COMBINATORIA)
    # epochs = config.EPOCHS 
    epochs = config.EPOCHS_COMBINATORIA 
    pop_size = config.POP_SIZE
    
    # ==========================================
    # CARGA DE FUNCIONES
    # ==========================================
    # ---- SE COMENTA Y DESCOMENTA SEGÚN LOS PROBLEMAS QUE SE VAYA A CORRER
    # functions = benchmark_CEC2017.functions
    # functions = problemas_ingenieria.problems 
    functions = tsp.problems
    ############################################
    # Bioinformatics Aplication Genes
    # X_train, X_test, y_train, y_test, best_k, selected_gens = leukemia_enviroment()
    # BioInstance = BioinformaticsFS(X_train, y_train, best_k)
    # functions = [BioInstance]
    ############################################
    # ==========================================
    # CONFIGURACIÓN ESTUDIO DE SENSIBILIDAD
    # ==========================================
    # ---- SE COMENTA Y DESCOMENTA SEGÚN LOS MODELOS QUE SE QUIERAN CORRER
    modelos_a_probar = modelos.models
    # modelos_a_probar = modelos.models_no_hybrids
    # modelos_a_probar = modelos.models + modelos.models_no_hybrids
    # modelos_a_probar = [{
    #     'modelo': HIBRIDO.hibrid_JADE,
    #     'nombre_carpeta': 'Hibrido_Markov_Extricto_Int5_Verdadero',
    #     'parametros': {'update_interval': 5, 'matriz_type': 'extricto', 'success_filter': True, 'memory_type': 'markov'}
    # }]
    
    for modelo_config in modelos_a_probar:
    
        # ==========================================
        # CONFIGURACIÓN PARA GUARDADO CSV
        # ==========================================
        Modelo_Clase = modelo_config['modelo']
        nombre_modelo = modelo_config['nombre_carpeta']
        parametros_modelo = modelo_config['parametros']
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        # ---- SE COMENTA Y DESCOMENTA SEGÚN SI ES CEC2017 U OTROS PROBLEMAS
        # folder_name = f"Resultados_{nombre_modelo}_{timestamp}_{dims}_dims" # <--- Se descomenta si se correrán funciones matemáticas y se comenta el otro
        folder_name = f"Resultados_{nombre_modelo}_{timestamp}" 
        # ---- SE COMENTA Y DESCOMENTA SEGÚN LOS PROBLEMAS QUE SE VAYA A CORRER
        # data_CEC = os.path.join(config.SENSIBILIDAD_DIR, 'CEC2017')
        # data_CEC = os.path.join(config.SENSIBILIDAD_DIR, 'CEC2017_Test_V2')
        # data_ING = os.path.join(config.SENSIBILIDAD_DIR, 'INGENIERIA')
        # data_ING = os.path.join(config.SENSIBILIDAD_DIR, 'INGENIERIA_Test_V2')
        # data_TSP = os.path.join(config.SENSIBILIDAD_DIR, 'TSP')
        data_TSP = os.path.join(config.SENSIBILIDAD_DIR, 'TSP_Test_V2')
        # data_BIO = os.path.join(config.RESULTS_APLICATIONS_DIR, 'Resultados_Bioinformatics')
        # Crear la carpeta físicamente
        # ---- SE COMENTA Y DESCOMENTA SEGÚN LOS PROBLEMAS QUE SE VAYA A CORRER
        # os.makedirs(data_CEC, exist_ok=True)
        # os.makedirs(data_ING, exist_ok=True)
        os.makedirs(data_TSP, exist_ok=True)
        # os.makedirs(data_BIO, exist_ok=True)
        # SE DESCOMENTA LA RUTA DE LA CARPETA QUE SE NECESITE (CEC, ING o TSP)
        # ---- SE COMENTA Y DESCOMENTA SEGÚN LOS PROBLEMAS QUE SE VAYA A CORRER
        # data_path = os.path.join(data_CEC, folder_name) 
        # data_path = os.path.join(data_ING, folder_name) 
        data_path = os.path.join(data_TSP, folder_name) 
        # data_path = os.path.join(data_BIO, folder_name)
        
        # Crear la carpeta físicamente
        os.makedirs(data_path, exist_ok=True)
        print(f">>> Carpeta de resultados creada en:\n {data_path}")
        
        # Nombre para el archivo de salida
        nombre_archivo_fitness      = os.path.join(data_path, "Fitness.csv")
        nombre_archivo_solution     = os.path.join(data_path, "Solucion.csv")
        nombre_archivo_tiempo       = os.path.join(data_path, "Tiempos.csv")
        nombre_archivo_convergencia = os.path.join(data_path, "Convergencia.csv")
        nombre_archivo_diversidad   = os.path.join(data_path, "Diversidad.csv")
        nombre_archivo_exploracion  = os.path.join(data_path, "Exploracion.csv")
        nombre_archivo_explotacion  = os.path.join(data_path, "Explotacion.csv")
        nombre_archivo_uso_modelos  = os.path.join(data_path, "Uso_Modelos.csv")
        
        # Diccionario para guardar TODOS los resultados crudos
        # Estructura: {'F1': [run1, run2...], 'F2': [run1, run2...]}
        raw_data = {} 
        raw_solution = {}
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
            func_name = function['name'] if isinstance(function, dict) else function.name
            print(f" -> Procesando función: {func_name}")
            
            its_bioinformatics = hasattr(function, 'get_metrics')
                        
            # Arreglos temporales para guardar los 30 resultados que irán llegando
            run_fitnesses = np.zeros(runs) # Lista temporal para los 30 fitness de ESTA función
            run_solutions = [None] * runs
            run_times = np.zeros(runs)
            fitness_per_epochs = np.zeros((runs, epochs))
            diversity_per_epochs = np.zeros((runs, epochs))
            exploration_per_epochs = np.zeros((runs, epochs))
            exploitation_per_epochs = np.zeros((runs, epochs))
            usage_DE_per_epochs = np.zeros((runs, epochs))
            usage_PSO_per_epochs = np.zeros((runs, epochs))
            usage_GA_per_epochs = np.zeros((runs, epochs))
            if its_bioinformatics:
                run_accuracies = np.zeros(runs)
                run_num_genes = np.zeros(runs)
            
            start_global = time.time()
            
            # --- INICIO DEL PROCESAMIENTO EN PARALELO ---
            # max_workers = Número de núcleos <------------------------------------
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
                        run_solutions[idx] = data['solution']
                        run_times[idx] = data['execution_time']
                        fitness_per_epochs[idx,:] = data['convergencia']
                        diversity_per_epochs[idx,:] = data['diversidad']
                        exploration_per_epochs[idx,:] = data['exploracion']
                        exploitation_per_epochs[idx,:] = data['explotacion']
                        usage_DE_per_epochs[idx,:] = data['uso_DE']
                        usage_PSO_per_epochs[idx,:] = data['uso_PSO']
                        usage_GA_per_epochs[idx,:] = data['uso_GA']
                        if its_bioinformatics:
                            run_accuracies = data['accuracy']
                            run_num_genes = data['num_genes']
                        
                        print(f"[OK] Corrida {idx+1} terminada - Fit: {data['fitness']:.4E}")
                    except Exception as e:
                        print(f"[ERROR] Una corrida falló: {e}")
                        
            end_global = time.time()
            print(f"\n¡Todas las corridas terminaron! Tiempo total global: {(end_global-start_global)/60:.2f} minutos")
            
            # --- AL TERMINAR LAS 30 CORRIDAS DE LA FUNCIÓN ---
        
            # 1. Guardar en el diccionario maestro (Esto es lo que irá al CSV)
            # Para los fitness y tiempo promedio por corrida
            raw_data[func_name] = run_fitnesses
            raw_solution[func_name] = run_solutions
            raw_times[func_name] = run_times
            # Para la convergencia, diversidad, exploracion y explotación por época (Trayectorias promedio)
            convergence_history[func_name] = np.mean(fitness_per_epochs, axis=0)
            raw_diversity[func_name] = np.mean(diversity_per_epochs, axis=0)
            raw_exploration[func_name] = np.mean(exploration_per_epochs, axis=0)
            raw_exploitation[func_name] = np.mean(exploitation_per_epochs, axis=0)
            raw_usage_models[f"{func_name}_DE"] = np.mean(usage_DE_per_epochs, axis=0)
            raw_usage_models[f"{func_name}_PSO"] = np.mean(usage_PSO_per_epochs, axis=0)
            raw_usage_models[f"{func_name}_GA"] = np.mean(usage_GA_per_epochs, axis=0)
                
            # 2. GUARDADO DE SEGURIDAD (Progressive Save)
            # Esto sobrescribe el archivo cada vez que termina una función.
            try:
                # Para el fitness
                guardar_csv(raw_data, nombre_archivo_fitness, 'Run_ID')
                # Para las soluciones
                guardar_csv(raw_solution, nombre_archivo_solution, 'Run_ID')
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
                
                if its_bioinformatics:
                    raw_accuracies = {func_name: run_accuracies}
                    raw_num_genes = {func_name: run_num_genes}
                    guardar_csv(raw_accuracies, os.path.join(data_path, 'Accuracy.csv'), 'Run_ID')
                    guardar_csv(raw_num_genes, os.path.join(data_path, 'Num_Genes.csv'), 'Run_ID')
                    
            except Exception as e:
                print(f"⚠️ Advertencia: No se pudo guardar el temporal ({e})")
            print("  >>> Guardado parcial exitoso.")
            
            # Cálculos estadísticos
            mean_fit = np.mean(run_fitnesses)
            std_fit = np.std(run_fitnesses)
            mean_time = np.mean(run_times)
            
            # Guardar para el reporte final (usando el nombre como clave)
            final_results[func_name] = {
                'mean': mean_fit,
                'best': np.min(run_fitnesses),
                'worst': np.max(run_fitnesses),
                'std': std_fit
            }
        
            # Reporte Individual
            print("-" * 50)
            print(f"RESUMEN: {func_name}")
            print(f"  Mejor : {np.min(run_fitnesses):.6E}")
            print(f"  Peor  : {np.max(run_fitnesses):.6E}")
            print(f"  Media : {mean_fit:.6E}")
            print(f"  Std   : {std_fit:.6E}")
            print(f"  Tiempo Promedio: {mean_time:.2f} s")
            # if its_bioinformatics:
            #     best_run_idx = np.argmin(run_fitnesses)
            #     best_solution = run_solutions[best_run_idx]
            #     acc_final, genes_seleccionados = get_final_values(best_solution, X_train, X_test, y_train, y_test, best_k, selected_gens)
            #     print(f"Genes seleccionados por la IA: {len(genes_seleccionados)}")
            #     print(f"Porcentaje de reducción: {100 - (len(genes_seleccionados)/X_test.shape[1]*100):.2f}%")
            #     print(f"Accuracy Final (Usando solo {len(genes_seleccionados)} genes): {acc_final * 100:.2f}%")
            #     print("\nTop 10 Genes más importantes descubiertos:")
            #     print(genes_seleccionados[:10])
            print("-" * 50)
      
    # ==========================================
    # GUARDADO DE CSV
    # ==========================================
    print("\n" + "="*60)
    print(">>> EXPERIMENTO FINALIZADO")