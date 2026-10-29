# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:38:14 2025

@author: osmio
"""

import numpy as np
from mealpy import Optimizer

class SPA(Optimizer):
    """
    This is an example how to build new optimizer
    """
    def __init__(self, epoch=2000, pop_size=25, k=10, vertex=3, atraction=0.15, convergence = 100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 50001])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [1, 500])
        self.k = self.validator.check_int("k", k, [2, 20])
        self.vertex = self.validator.check_int("vertex", vertex, [3, 10])
        self.atraction = self.validator.check_float("atraction", atraction, [0, 1])
        self.convergence = self.validator.check_int("vertex", vertex, [1, 1E8])
        self.sort_flag = False 
    
    def initialize_variables(self):
       self.pv = np.zeros((self.pop_size, self.problem.n_dims+1))
       self.pvhist = np.zeros((self.pop_size, self.k, self.problem.n_dims+1))
       
       self.space = self.problem.ub - self.problem.lb
    
    def initialization(self):
        """
        Override this method if needed. But the first 2 lines of code is required.
        """
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        
        self.pvhist[:, :, 0] = np.inf
        
        for idx in range(self.pop_size):
            self.pvhist[idx, 0, 1:] = self.pop[idx].solution
            self.pvhist[idx, 0, 0] = self.pop[idx].target.fitness
            #############
            self.pv[idx, 0] = self.pop[idx].target.fitness
            self.pv[idx, 1:] = self.pop[idx].solution
            #############
                
              
    def evolve(self, epoch):
        """
        You can do everything in this function (i.e., Loop through the population multiple times)

        Args:
            epoch (int): The current iteration
        """      
        # Inicialización de los pelotones
        if epoch<self.k:
            for idx in range(self.pop_size):
                pos_new = np.random.normal(self.pv[idx, 1:], scale=self.space/6)
                pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
                agent = self.generate_agent(pos_new)
                self.pop[idx] = agent
                
                self.pvhist[idx, epoch%self.k, 1:] = self.pop[idx].solution
                self.pvhist[idx, epoch%self.k, 0] = self.pop[idx].target.fitness
        
        #Trabajo en estado normal
        else:
            #Para cada uno de los pelotones
            for idx in range(self.pop_size):
                # Se evaluan los trabajadores
                index = np.argsort(self.pvhist[idx, :, 0])
                # Se seleccionan los mejores 3
                #se extrae su fitness promedio y posicion promedio
                self.pv[idx, 0] = self.pvhist[idx, index[:self.vertex], 0].mean()
                self.pv[idx, 1:] = self.pvhist[idx, index[:self.vertex], 1:].mean(axis=0)
            
            #Para toda la poblacion 
            #Se extraen los mejores 3 pelotones y se genera con su posicion promedio
            #la el nuevo punto de interes
            index = np.argsort(self.pv[:, 0])
            mv = self.pv[index[:3], 1:].mean(axis=0)
            
            # Para cada peloton
            for idx in range(self.pop_size):
                #Si no es el peor peloton
                if idx!=index[-1]:
                    #Se ajusta el punto virtual de cada peloton para dirigirse
                    #al punto de interes global
                    self.pv[idx, 1:] = mv*self.atraction + (1-self.atraction)*self.pv[idx, 1:]
                #Caso contrario
                else:
                    #Con un 10% de probabilidad
                    r = np.random.uniform()
                    if r<0.1:
                        # Se genera un oposition
                        self.pv[idx, 1:] = mv*(2-self.atraction) - (1-self.atraction)*self.pv[idx, 1:]
                    else:
                        # O una atraccion simple
                        self.pv[idx, 1:] = mv*self.atraction + (1-self.atraction)*self.pv[idx, 1:]
            
            #Para cada peloton
            for idx in range(self.pop_size):
                
                #x = (np.log10(epoch)/np.log10(4))
                # Empleando esta convergencia
                y = epoch/self.convergence + 1
                # Realiza una perturvacion de la particula virtual
                pos_new = np.random.normal(self.pv[idx, 1:], scale=self.space/6/y)
                #self.space = self.space * 0.99
                #pos_new = np.random.normal(self.pv[idx, 1:], scale=self.space)
                pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
                agent = self.generate_agent(pos_new)
                self.pop[idx] = agent
                
                self.pvhist[idx, epoch%self.k, 1:] = self.pop[idx].solution
                self.pvhist[idx, epoch%self.k, 0] = self.pop[idx].target.fitness