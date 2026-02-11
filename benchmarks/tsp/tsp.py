# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 19:15:42 2026

@author: oswal
"""
import tsplib95
import numpy as np
import os

class TSPInstance:
    def __init__(self, filepath):
        """
        Carga el problema TSP y prepara los datos.
        """
        # Cargamos el problema
        self.problem = tsplib95.load(filepath)
        # Extraemos datos relevantes para main
        self.name = self.problem.name
        self.dimension = self.problem.dimension
        # Obtenemos la lista de nodos 
        self.nodes = list(self.problem.get_nodes())
        # Definimos los límites inferior y superior 
        self.lb = [0.0]*self.dimension
        self.ub = [1.0]*self.dimension
    def evaluate(self, solution):
        """
        Función de Fitness usando RANDOM KEYS.
        Convierte el vector continuo (floats) en una ruta (permutación).
        """
        # Obtenemos la ruta
        ruta = np.argsort(solution)
        # Calculamos la distancia total recorriendo la ruta
        distancia_total = 0
        for i in range(len(ruta)-1):
            u = self.nodes[ruta[i]]
            v = self.nodes[ruta[i+1]]
            # La librería calcula la distancia (geográfica o euclidiana)
            distancia_total += self.problem.get_weight(u, v)
        # Sumar el regreso al inicio
        u = self.nodes[ruta[-1]]
        v = self.nodes[ruta[0]]
        distancia_total += self.problem.get_weight(u, v)
        
        return distancia_total
        
# =================================================================
# LISTA DE PROBLEMAS CARGADOS
# =================================================================

base_path = os.path.dirname(os.path.abspath(__file__))

files = [
    'burma14.tsp',
    'bayg29.tsp',
    'att48.tsp',
    'berlin52.tsp',
    'eil76.tsp'
]

problems = []
for f in files:
    full_path = os.path.join(base_path, f)
    if os.path.exists(full_path):
        problems.append(TSPInstance(full_path))
    else:
        print(f"⚠️ Advertencia: No se encontró {f} en {base_path}")