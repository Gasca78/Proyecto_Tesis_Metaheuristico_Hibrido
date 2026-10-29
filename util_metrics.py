# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:47:43 2026

@author: oswal
"""
import numpy as np

def avg_dist_gb(pop, g_best):
    """Calcula la distancia media de la población respecto al global best."""
    distances = []
    for agent in pop:
        dist = np.linalg.norm(agent.solution - g_best.solution)
        distances.append(dist)
    return np.mean(distances)

def fitness_diversity(pop, e=1e-10):
    """Calcula la diversidad del fitness de la población."""
    fitness_values = np.array([agent.target.fitness for agent in pop])
    std_fit = np.std(fitness_values)
    range_fit = np.max(fitness_values) - np.min(fitness_values)
    return std_fit / (range_fit + e)