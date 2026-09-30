# -*- coding: utf-8 -*-
"""
Created on Tue Sep 8 10:55:25 2026

@author: oswal
"""

from HIBRIDO import hibrid_JADE

# --- FAMILIA: MARKOV CONSERVADOR ---

class hibrid_JADE_markov_conservador_5_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        # Llamamos al __init__ del padre forzando los parámetros que queremos
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            matriz_type='conservador', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        # Opcional pero súper útil: Esto hace que Mealpy imprima el nombre bonito en la consola
        self.name = "Hibrido_Markov_Conservador_Int5_Falso"
        
class hibrid_JADE_markov_conservador_5_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        # Llamamos al __init__ del padre forzando los parámetros que queremos
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            matriz_type='conservador', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        # Opcional pero súper útil: Esto hace que Mealpy imprima el nombre bonito en la consola
        self.name = "Hibrido_Markov_Conservador_Int5_Verdadero"
        
class hibrid_JADE_markov_conservador_10_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            matriz_type='conservador', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Conservador_Int10_Falso"
        
class hibrid_JADE_markov_conservador_10_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            matriz_type='conservador', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Conservador_Int10_Verdadero"
        
class hibrid_JADE_markov_conservador_50_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            matriz_type='conservador', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Conservador_Int50_Falso"
        
class hibrid_JADE_markov_conservador_50_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            matriz_type='conservador', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Conservador_Int50_Verdadero"
        
# --- FAMILIA: MARKOV MODERADO ---

class hibrid_JADE_markov_moderado_5_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        # Llamamos al __init__ del padre forzando los parámetros que queremos
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            matriz_type='moderado', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        # Opcional pero súper útil: Esto hace que Mealpy imprima el nombre bonito en la consola
        self.name = "Hibrido_Markov_Moderado_Int5_Falso"
        
class hibrid_JADE_markov_moderado_5_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        # Llamamos al __init__ del padre forzando los parámetros que queremos
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            matriz_type='moderado', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        # Opcional pero súper útil: Esto hace que Mealpy imprima el nombre bonito en la consola
        self.name = "Hibrido_Markov_Moderado_Int5_Verdadero"
        
class hibrid_JADE_markov_moderado_10_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            matriz_type='moderado', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Moderado_Int10_Falso"
        
class hibrid_JADE_markov_moderado_10_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            matriz_type='moderado', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Moderado_Int10_Verdadero"
        
class hibrid_JADE_markov_moderado_50_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            matriz_type='moderado', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Moderado_Int50_Falso"
        
class hibrid_JADE_markov_moderado_50_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            matriz_type='moderado', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Moderado_Int50_Verdadero"
        
# --- FAMILIA: MARKOV ESTRICTO ---

class hibrid_JADE_markov_estricto_5_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        # Llamamos al __init__ del padre forzando los parámetros que queremos
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            matriz_type='estricto', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        # Opcional pero súper útil: Esto hace que Mealpy imprima el nombre bonito en la consola
        self.name = "Hibrido_Markov_Estricto_Int5_Falso"
        
class hibrid_JADE_markov_estricto_5_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        # Llamamos al __init__ del padre forzando los parámetros que queremos
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            matriz_type='estricto', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        # Opcional pero súper útil: Esto hace que Mealpy imprima el nombre bonito en la consola
        self.name = "Hibrido_Markov_Estricto_Int5_Verdadero"
        
class hibrid_JADE_markov_estricto_10_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            matriz_type='estricto', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Estricto_Int10_Falso"
        
class hibrid_JADE_markov_estricto_10_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            matriz_type='estricto', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Estricto_Int10_Verdadero"
        
class hibrid_JADE_markov_estricto_50_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            matriz_type='estricto', 
            success_filter=False, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Estricto_Int50_Falso"
        
class hibrid_JADE_markov_estricto_50_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            matriz_type='estricto', 
            success_filter=True, 
            memory_type='markov', 
            **kwargs
        )
        self.name = "Hibrido_Markov_Estricto_Int50_Verdadero"

# --- FAMILIA: PROBABILIDADES ---

class hibrid_JADE_probabilidades_5_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            success_filter=False, 
            memory_type='probs', 
            **kwargs
        )
        self.name = "Hibrido_Probabilidades_Int5_Falso"
        
class hibrid_JADE_probabilidades_5_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=5, 
            success_filter=True, 
            memory_type='probs', 
            **kwargs
        )
        self.name = "Hibrido_Probabilidades_Int5_Verdadero"
        
class hibrid_JADE_probabilidades_10_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            success_filter=False, 
            memory_type='probs', 
            **kwargs
        )
        self.name = "Hibrido_Probabilidades_Int10_Falso"
        
class hibrid_JADE_probabilidades_10_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=10, 
            success_filter=True, 
            memory_type='probs', 
            **kwargs
        )
        self.name = "Hibrido_Probabilidades_Int10_Verdadero"
        
class hibrid_JADE_probabilidades_50_F(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            success_filter=False, 
            memory_type='probs', 
            **kwargs
        )
        self.name = "Hibrido_Probabilidades_Int50_Falso"
        
class hibrid_JADE_probabilidades_50_V(hibrid_JADE):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(
            epoch=epoch, 
            pop_size=pop_size, 
            update_interval=50, 
            success_filter=True, 
            memory_type='probs', 
            **kwargs
        )
        self.name = "Hibrido_Probabilidades_Int50_Verdadero"