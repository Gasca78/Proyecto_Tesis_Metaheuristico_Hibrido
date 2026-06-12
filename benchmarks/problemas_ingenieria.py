# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 16:38:58 2026

@author: oswal
"""

from enoppy.paper_based import rwco_2020, pdo_2022

# Lista de problemas a probar
# problems = [
#     # RWCO_2020
#     rwco_2020.HeatExchangerNetworkDesignCase1Problem(),    # p1
#     rwco_2020.HeatExchangerNetworkDesignCase2Problem(),    # p2
#     rwco_2020.HaverlyPoolingProblem(),                     # p3
#     rwco_2020.BlendingPoolingSeparationProblem(),          # p4
#     rwco_2020.PropaneIsobutaneNButaneNonsharpSeparationProblem(), # p5
#     rwco_2020.OptimalOperationAlkylationUnitProblem(),     # p6
#     rwco_2020.ReactorNetworkDesignProblem(),               # p7
#     rwco_2020.ProcessSynthesis01Problem(),                 # p8
#     rwco_2020.ProcessSynthesisAndDesignProblem(),          # p9
#     rwco_2020.ProcessFlowSheetingProblem(),                # p10
#     rwco_2020.TwoReactorProblem(),                         # p11
#     rwco_2020.ProcessSynthesis02Problem(),                 # p12
#     rwco_2020.ProcessDesignProblem(),                      # p13
#     rwco_2020.MultiProductBatchPlantProblem(),             # p14
#     rwco_2020.WeightMinimizationSpeedReducerProblem(),     # p15
#     rwco_2020.OptimalDesignIndustrialRefrigerationSystemProblem(), # p16
#     rwco_2020.TensionCompressionSpringDesignProblem(),     # p17
#     rwco_2020.PressureVesselDesignProblem(),               # p18
#     rwco_2020.WeldedBeamDesignProblem(),                   # p19
#     rwco_2020.ThreeBarTrussDesignProblem(),                # p20
#     rwco_2020.MultipleDiskClutchBrakeDesignProblem(),      # p21
#     rwco_2020.PlanetaryGearTrainDesignOptimizationProblem(), # p22
#     rwco_2020.StepConePulleyProblem(),                      # p23
#     # PDO 2022
#     pdo_2022.CantileverBeamProblem(),       # p24
#     pdo_2022.IBeamProblem(),                # p25
#     pdo_2022.TubularColumnProblem(),        # p26
#     pdo_2022.PistonLeverProblem(),          # p27
#     pdo_2022.CorrugatedBulkheadProblem(),   # p28
#     pdo_2022.ReinforcedConcreateBeamProblem(), # p29
#     pdo_2022.GearTrainProblem()             # p30
# ]
problems = [
    # RWCO_2020
    rwco_2020.WeightMinimizationSpeedReducerProblem(),     # p15
    rwco_2020.TensionCompressionSpringDesignProblem(),     # p17
    rwco_2020.PressureVesselDesignProblem(),               # p18
    rwco_2020.WeldedBeamDesignProblem(),                   # p19
    rwco_2020.ThreeBarTrussDesignProblem()                # p20
]