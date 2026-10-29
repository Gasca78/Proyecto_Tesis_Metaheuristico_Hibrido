# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:37:49 2024

@author: osmio
"""

import numpy as np
from mealpy import Optimizer, FloatVar
from mealpy.utils.agent import Agent
from scipy.interpolate import lagrange

class ETO(Optimizer):
    """
    This is an example how to build new optimizer
    """
#self, epoch=1000, pop_size=50, pl=0.1, conv=0.25, **kwargs
    def __init__(self, epoch=1000, pop_size=50, pl=0.1, conv=0.25, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [1, 10000])
        self.pl = self.validator.check_float("Percentage of linterns", pl, [0,1])
        self.conv = self.validator.check_float("Percentage of convergence", conv, [0,1])
        
        self.sort_flag = True

    def initialize_variables(self):
        self.g = 30
        
    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
       
        self.ss = self.problem.ub - self.problem.lb
        
    def refine(self, idx):
        fitness = np.array([x.target.fitness for x in self.pop])
        index = np.argsort(fitness)[1]
               
        r1 = np.random.uniform(0, 2)
        r2 = np.power(np.random.uniform(0, 1),1/2)*1.5
        
        pos_new = r2*(r1*self.g_best.solution + (1-r1)*self.pop[index].solution)  + (1-r2)*self.pop[idx].solution
        
        return pos_new
        
    def explore(self, idx, ran, epoch):
        if epoch%25 == 0:
            #pos_new = (self.pop[idx-1].solution - self.g_best.solution)*5 + self.g_best.solution
            pos_new = self.ss*np.random.uniform(-0.5,0.5)/np.log10(epoch) + self.g_best.solution
            
            return pos_new
        else:
            d1 = np.random.randint(0, self.problem.n_dims)
            d2 = np.random.randint(0, self.problem.n_dims)
            while d1 == d2:
                d2 = np.random.randint(0, self.problem.n_dims)
            
            delta1 = self.pop[idx].solution[d1] - self.g_best.solution[d1]
            delta2 = self.pop[idx].solution[d2] - self.g_best.solution[d2]
            tem1 = delta1*np.cos(np.radians(self.g))-delta2*np.sin(np.radians(self.g))
            tem2 = delta2*np.cos(np.radians(self.g))+delta1*np.sin(np.radians(self.g))
            
            r = np.random.normal(0, 0.3)
            
            tem1 = tem1*(self.conv+r) + self.g_best.solution[d1]
            tem2 = tem2*(self.conv+r) + self.g_best.solution[d2]
            
            pos_new = self.pop[idx].solution
            pos_new[d1] = tem1
            pos_new[d2] = tem2
            return pos_new
            
    def evolve(self, epoch):
               
        for idx in range(self.pop_size):
            if idx > self.pop_size*(1-self.pl):
                pos_new = self.refine(idx)
                pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
                agent = self.generate_agent(pos_new)
                self.pop[idx] = agent
            
            else:
                fitness = np.array([x.target.fitness for x in self.pop])
                index = np.argsort(fitness)
                r = np.random.choice([0,1,2])
                ran = index[r]
                
                pos_new = self.explore(idx, ran, epoch)
                pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
                agent = self.generate_agent(pos_new)
                self.pop[idx] = agent