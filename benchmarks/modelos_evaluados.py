# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 23:54:49 2026

@author: oswal
"""

from algoritmos import ETO, SPA
from mealpy import DE, SHADE, PSO, GA, GWO, SLO
from util_metrics import avg_dist_gb, fitness_diversity

# 1. Clase que inyecta la medición
class EvaluadorMetricasDistancias:
    def init_metrics_tracker(self):
        """Inicializa los arreglos para guardar el historial."""
        self.db = []
        self.fd = []

    def update_metrics(self):
        """Calcula y guarda las métricas de la época actual."""
        # pop y g_best son atributos estándar nativos en todos los modelos de Mealpy
        self.db.append(avg_dist_gb(self.pop, self.g_best))
        self.fd.append(fitness_diversity(self.pop))

# 2. Modelos Envueltos (Original + Mixin)
class PSO_Metrics(PSO.OriginalPSO, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch) # Corre la época normal de PSO
        self.update_metrics() # Toma la "fotografía" de la métrica
        
class P_PSO_Metrics(PSO.P_PSO, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics() 

class JADE_Metrics(DE.JADE, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()
        
class SADE_Metrics(DE.SADE, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()

class GA_Metrics(GA.BaseGA, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()
        
class ISLO_Metrics(SLO.ImprovedSLO, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()
        
class ETO_Metrics(ETO.ETO, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()
        
class SPA_Metrics(SPA.SPA, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()
        
#### NO USADOS EN EXPERIMENTACIÓN ARTICULO
        
class SHADE_Metrics(SHADE.OriginalSHADE, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()

class GWO_Metrics(GWO.OriginalGWO, EvaluadorMetricasDistancias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_metrics_tracker()

    def evolve(self, epoch):
        super().evolve(epoch)
        self.update_metrics()