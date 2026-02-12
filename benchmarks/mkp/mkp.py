# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:25:08 2026

@author: oswal
"""

import numpy as np
import os

class MKPInstance():
    def __init__(self, num_items, num_constraints, optimal, profits, constraints, capacities, name):
        self.name = name
        self.n = num_items # Número de objetos
        self.m = num_constraints # Número de restricciones
        self.optimal = optimal # Valor óptimo conocido
        
        # Convertimos a arrays de numpy para velocidad
        self.profits = np.array(profits) # Vector de Ganancias (p)
        self.constraints = np.array(constraints) # Matriz de Pesos (Axn)
        self.capacities = np.array(capacities) # Vector de Capacidades (b)
        
        # Límites (Continuos entre 0 y 1)
        self.lb = [0.0] * self.n
        self.ub = [1.0] * self.n
        self.dimension = self.n # Para el main.py
        
    def evaluate(self, solution):
        """
        Función de Fitness con Binarización Estática.
        1. Convierte continuo -> binario (Umbral 0.5).
        2. Verifica restricciones.
        3. Si es factible: Retorna -Ganancia (porque minimizamos).
        4. Si es infactible: Retorna penalización.
        """
        # 1. Binarización (Transfer Function: Step)
        # Si el valor obtenido por el modelo es > 0.5, seleccionamos el objeto (1), sino (0)
        binary_selection = (np.array(solution) > 0.5).astype(int)
        # Si no se seleccionó nada, retornamos 0
        if np.sum(binary_selection) == 0:
            return 0
        
        # 2. Calcular ganancia total (Profit)
        total_profit = np.sum(binary_selection*self.profits)
        
        # 3. Verificar Restricciones (Matricial es más rápido)
        # Multiplicamos la matriz de pesos por el vector de selección
        # Resultado: Cuánto pesa la mochila en cada una de las M dimensiones
        used_resources = np.dot(self.constraints, binary_selection)
        
        # Verificamos si algún recurso excede su capacidad
        if np.any(used_resources > self.capacities):
            # La mochila se rompió
            # Penalización: Retornar un valor positivo gigante 
            return 1e9
        
        # 4. Retorno (Factible)
        # Se regresa negativo, entre más negativo, más ganancia ya que se minimiza
        return -total_profit
    
def load_mkp_file(filepath):
    """
    Parser robusto para leer los archivos de OR-Library MKP.
    Retorna una lista de objetos MKPInstance.
    """
    problems = []
    filename = os.path.basename(filepath)
    
    with open(filepath, 'r') as f:
        # Leemos todo el contenido y lo convertimos en una lista plana de números
        # Esto soluciona el problema de saltos de línea irregulares
        raw_data = f.read().split()
        
    iterator = iter(raw_data)
    
    try:
        num_problems = int(next(iterator)) # El primer número es la cantidad de problemas
        
        for i in range(num_problems):
            # Cabecera del problema
            n = int(next(iterator)) # Items
            m = int(next(iterator)) # Restricciones
            opt = float(next(iterator)) # Óptimo
            
            # Leer Ganancias (n items)
            profits = [float(next(iterator)) for _ in range(n)]
            
            # Leer Restricciones (m filas * n columnas)
            # En el archivo vienen m bloques de n números
            constraints = []
            for _ in range(m):
                row = [float(next(iterator)) for _ in range(n)]
                constraints.append(row)
            
            # Leer Capacidades (m restricciones)
            capacities = [float(next(iterator)) for _ in range(m)]
            
            # Crear objeto
            prob_name = f"{filename}_P{i+1}_{n}items"
            problems.append(MKPInstance(n, m, opt, profits, constraints, capacities, prob_name))
    except StopIteration:
        pass
    
    return problems
    
# =================================================================
# CARGA AUTOMÁTICA
# =================================================================
base_path = os.path.dirname(os.path.abspath(__file__))

file_target = os.path.join(base_path, 'mknap1.txt')

problems = []
if os.path.exists(file_target):
    problems = load_mkp_file(file_target)
else:
    print(f"⚠️ No se encontró {file_target}")