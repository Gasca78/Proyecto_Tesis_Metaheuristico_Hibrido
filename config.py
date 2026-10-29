import os

# Parámetros del Algoritmo
DIMS_30 = 30
DIMS_50 = 50
DIMS_100 = 100
RUNS = 30
EPOCHS = 1000
POP_SIZE = 100
# EPOCHS = 500
# POP_SIZE = 50
EPOCHS_COMBINATORIA = 1000

# Rutas (Hacerlas relativas para que funcionen en cualquier compu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados') # Todo se guardará en la carpeta 'results'
SENSIBILIDAD_DIR = os.path.join(BASE_DIR, 'Estudio_de_Sensibilidad') # Aquí se guardará el estudio de sensibilidad
RESULTS_APLICATIONS_DIR = os.path.join(BASE_DIR, 'Aplicaciones_CD')

# Crear la carpeta de resultados si no existe (buena práctica)
os.makedirs(RESULTS_DIR, exist_ok=True)