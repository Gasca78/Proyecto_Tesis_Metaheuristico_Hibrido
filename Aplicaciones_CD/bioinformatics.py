# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:04:20 2026

@author: oswal
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

def leukemia_enviroment():
    """Descarga los datos, hace el Baseline y GridSearch"""
    print(">>> Preparando Entorno Bioinformático...")
    dataset = fetch_openml(name="leukemia", version=1, parser="auto")
    X = dataset.data.values
    y = dataset.target.values
    selected_gens = np.array(dataset.feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # GridSearch rápido
    parametros_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    grid = GridSearchCV(KNeighborsClassifier(), parametros_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    best_k = grid.best_params_['n_neighbors']
    print(f">>> GridSearch completado. Mejores valores: {grid.best_params_}")
    
    return X_train, X_test, y_train, y_test, best_k, selected_gens

class BioinformaticsFS:
    def __init__(self, X, y, best_k, alpha=0.99, beta=0.01):
        self.name = "Leukemia_Feature_Selection"
        self.X = X
        self.y = y
        self.num_features = X.shape[1]
        self.lb = [0.0] * self.num_features
        self.ub = [1.0] * self.num_features
        self.alpha = alpha
        self.beta = beta
        self.best_k = best_k

    def evaluate(self, solution):
        # Binarizar el vector de la solución (0 a 1)
        mask = np.array(solution) > 0.5
        num_selected = np.sum(mask)

        if num_selected == 0:
            return 1.0 # Castigo máximo si apaga todos los genes

        X_selected = self.X[:, mask]

        # Evaluamos con el mejor K que encontró GridSearchCV (ej. k=3 o k=5)
        knn = KNeighborsClassifier(n_neighbors=self.best_k)

        scores = cross_val_score(knn, X_selected, self.y, cv=5, scoring='accuracy')
        error_rate = 1.0 - np.mean(scores)

        feature_ratio = num_selected / self.num_features
        fitness = (self.alpha * error_rate) + (self.beta * feature_ratio)

        return fitness
    
    def get_metrics(self, solution):
        mask = np.array(solution) > 0.5
        num_selected = np.sum(mask)
        if num_selected == 0:
            return {'accuracy': 0.0, 'num_genes': 0}
            
        X_selected = self.X[:, mask]
        knn = KNeighborsClassifier(n_neighbors=self.best_k)
        scores = cross_val_score(knn, X_selected, self.y, cv=5, scoring='accuracy')
        
        return {'accuracy': np.mean(scores), 'num_genes': num_selected}