# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 14:10:05 2025

@author: oswal
"""

#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from scipy.stats import cauchy
# from mealpy.utils.agent import Agent

class hibrid_JADE(Optimizer):
    """
    The original version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DE
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = DE.JADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, miu_f: float = 0.5,
                 miu_cr: float = 0.5, pt: float = 0.1, ap: float = 0.1, 
                 c1: float = 2.05, c2: float = 2.05, w: float = 0.4, # Variables PSO
                 pc: float = 0.95, pm: float = 0.025, # Variables GA
                 **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial adaptive f, default = 0.5
            miu_cr (float): initial adaptive cr, default = 0.5
            pt (float): The percent of top best agents (p in the paper), default = 0.1
            ap (float): The Adaptation Parameter control value of f and cr (c in the paper), default=0.1
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
            PSO
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
            GA
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
            Hibridación
            strategies_probs (float): array with probabilites of each model, default=[1/3, 1/3, 1/3]
            strategies_rewards (float): array with rewards for each model, default=[0.0, 0.0, 0.0]
            strategies_usage (int): counter of uses for each mdodel, default=[0, 0, 0]
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        # np.random.uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.pt = self.validator.check_float("pt", pt, (0, 1.0))
        # np.random.uniform(1/20, 1/5) # the adaptation parameter control value of f and cr
        self.ap = self.validator.check_float("ap", ap, (0, 1.0))
        # Sección variables iniciadoras en PSO
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w = self.validator.check_float("w", w, (0, 1.0))
        # Sección variables iniciadoras en GA
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True
        # Seteado de parámetros
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr", "pt", "ap", # Variables JADE ("DE") 
                             "c1", "c2", "w", # Variables PSO
                             "pc", "pm"]) # Variables GA 
        self.sort_flag = False
        # Variables para Hibridación
        self.strategies_probs = [1/3, 1/3, 1/3]
        # Acumuladores para ver qué tanto funcionó cada uno
        self.strategies_rewards = {'DE':0.0, 'PSO':0.0, 'GA':0.0}
        self.strategies_usage = {'DE':0, 'PSO':0, 'GA':0}
        # Implementación de Cadenas de Markov
        self.transition_matriz = [[0.90,0.05,0.05],[0.05,0.90,0.05],[0.05,0.05,0.90]]
        # self.best_model = ''

    def initialize_variables(self):
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = list()
        
    ### Survivor Selection
    def lehmer_mean(self, list_objects):
        temp = np.sum(list_objects)
        return 0 if temp == 0 else np.sum(list_objects ** 2) / temp
    
    ### Función para agregar atributos de velocidad, posición local y fitness local a cada agente
    def PSO_initiation(self):
        # Inicialización de variables PSO
        for agent in self.pop:
            if not hasattr(agent, 'velocity'):
                agent.velocity = np.zeros(self.problem.n_dims)
            if not hasattr(agent, 'local_solution'):    
                agent.local_solution = agent.solution.copy()
            if not hasattr(agent, 'local_target'):
                agent.local_target = agent.target.copy()

    
    ### PSO model
    def PSO(self, idx):
        cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution)
        social = self.c2 * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
        self.pop[idx].velocity = self.w * self.pop[idx].velocity + cognitive + social
        x_new = self.pop[idx].solution + self.pop[idx].velocity
        return self.correct_solution(x_new)
    
    ### GA model
    def GA(self, idx):
        # Crossover
        r1_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
        
        current_pos = self.pop[idx].solution
        partner_pos = self.pop[r1_idx].solution
        
        if self.generator.random() < self.pc:
            alpha = self.generator.random() # Crea un valor entre 0 y 1
            x_new = current_pos * alpha + partner_pos * (1-alpha)
        else:
            x_new = current_pos.copy()
            
        # Mutation
        mutation = self.generator.normal(0, 0.1, size=self.problem.n_dims)
        x_new = x_new + mutation
        return self.correct_solution(x_new)
    
    ### Función para calcular y actualizar los porcentajes para los modelos
    def update_strategies_probabilities(self):
        qualities = np.array([]) # Arreglo para calcular la mejora de calidad por modelo
        total_quality = 0.0 # Para calcular la mejora total
        epsilon = 1e-10 # Para evitar diviones por 0
        for model in ['DE','PSO','GA']:
            if self.strategies_usage[model] > 0:
                quality = self.strategies_rewards[model]/self.strategies_usage[model]
            else:
                quality = epsilon
            qualities = np.append(qualities, quality)
            total_quality += quality
        
        if total_quality == 0:
            self.strategies_probs = [1/3, 1/3, 1/3]
        else:
            prob_DE = qualities[0]/total_quality
            prob_PSO = qualities[1]/total_quality
            prob_GA = 1 - (prob_DE+prob_PSO)
            if prob_GA < 0: prob_GA = 0.0
            self.strategies_probs = [prob_DE, prob_PSO, prob_GA]
            
        # Configuración de las probs en Markov chain
        self.best_model = np.argmax(qualities)
        if self.best_model == 0:
            self.transition_matriz = [[0.90,0.05,0.05],[0.90,0.05,0.05],[0.90,0.05,0.05]]
        elif self.best_model == 1:
            self.transition_matriz = [[0.05,0.90,0.05],[0.05,0.90,0.05],[0.05,0.90,0.05]]
        else:
            self.transition_matriz = [[0.05,0.05,0.90],[0.05,0.05,0.90],[0.05,0.05,0.90]]
            
        self.strategies_rewards = {'DE':0.0, 'PSO':0.0, 'GA':0.0}
        self.strategies_usage = {'DE':0, 'PSO':0, 'GA':0}
    
    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        if epoch == 1 or not hasattr(self.pop[0], 'velocity'):
            self.PSO_initiation()
        list_f = list()
        list_cr = list()
        temp_f = list()
        temp_cr = list()
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            # Se elige el método para este agente
            if epoch == 1:
                strategy_idx = self.generator.choice([0,1,2], p=self.strategies_probs)
            else:
                previus_model = self.pop[idx].model
                if previus_model == 'PSO':
                    strategy_idx = self.generator.choice([0,1,2], p=self.transition_matriz[1])
                elif previus_model == 'GA':
                    strategy_idx = self.generator.choice([0,1,2], p=self.transition_matriz[2])
                else:
                    strategy_idx = self.generator.choice([0,1,2], p=self.transition_matriz[0])
            # Uso para cada método
            if strategy_idx == 1:
                # Usamos el método de PSO
                self.pop[idx].model = 'PSO'
                x_new = self.PSO(idx)
                temp_f.append(0.5)
                cr = 0.9
                temp_cr.append(cr)
            elif strategy_idx == 2:
                # Usamos el método de GA
                self.pop[idx].model = 'GA'
                x_new = self.GA(idx)
                temp_f.append(0.5)
                cr = 0.9
                temp_cr.append(cr)
            else:
                # Usamos el método de JADE
                self.pop[idx].model = 'DE'
                ## Calculate adaptive parameter cr and f
                cr = self.generator.normal(self.dyn_miu_cr, 0.1)
                cr = np.clip(cr, 0, 1)
                while True:
                    f = cauchy.rvs(self.dyn_miu_f, 0.1)
                    if f < 0:
                        continue
                    elif f > 1:
                        f = 1
                    break
                temp_f.append(f)
                temp_cr.append(cr)
                top = int(self.pop_size * self.pt)
                x_best = pop_sorted[self.generator.integers(0, top)]
                r1_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                new_pop = self.pop + self.dyn_pop_archive
                r2_idx = self.generator.choice(list(set(range(0, len(new_pop))) - {idx, r1_idx}))
                x_r1 = self.pop[r1_idx].solution
                x_r2 = new_pop[r2_idx].solution
                x_new = self.pop[idx].solution + f * (x_best.solution - self.pop[idx].solution) + f * (x_r1 - x_r2)      
            pos_new = np.where(self.generator.random(self.problem.n_dims) < cr, x_new, self.pop[idx].solution)
            j_rand = self.generator.integers(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.model = self.pop[idx].model
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        pop = self.update_target_for_population(pop)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop[idx].target, self.pop[idx].target, self.problem.minmax):
                self.dyn_pop_archive.append(self.pop[idx].copy())
                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])
                # Lógica para actualización de porcentajes de los modelos
                # Calculamos cuánto mejoró
                improvement = np.abs(self.pop[idx].target.fitness - pop[idx].target.fitness)
                self.strategies_rewards[self.pop[idx].model] += improvement
                self.strategies_usage[self.pop[idx].model] += 1 # Sumamos 1 a la estrategia utilizada
                # Espacio para pasar atributos de PSO
                pop[idx].velocity = self.pop[idx].velocity.copy() # Si o si pasamos velocidad al hijo
                # Si se usa "PSO", se debe de actualizar la "Memoria Personal"
                if self.pop[idx].model == 'PSO':
                    pop[idx].local_solution = pop[idx].solution.copy()
                    pop[idx].local_target = pop[idx].target.copy()
                else: # De no usar PSO, se heredará las características del padre
                    pop[idx].local_solution = self.pop[idx].local_solution.copy()
                    pop[idx].local_target = self.pop[idx].local_target.copy()
                self.pop[idx] = pop[idx].copy()
        # ACTUALIZACIÓN de Probabilidades (cada 10 épocas)
        if epoch % 10 == 0:
            self.update_strategies_probabilities()
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, solution in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(solution)
            self.dyn_pop_archive = archive_pop_new
        # Update miu_cr and miu_f
        if len(list_cr) == 0:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * 0.5
        else:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * np.mean(np.array(list_cr))
        if len(list_f) == 0:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * 0.5
        else:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * self.lehmer_mean(np.array(list_f))
