# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:38:58 2026

@author: oswal
"""

from enoppy.paper_based import rwco_2020

# Lista de problemas a probar
problems = [
    rwco_2020.WeightMinimizationSpeedReducerProblem(),
    rwco_2020.TensionCompressionSpringDesignProblem(),
    rwco_2020.PressureVesselDesignProblem(),
    rwco_2020.WeldedBeamDesignProblem()
]
