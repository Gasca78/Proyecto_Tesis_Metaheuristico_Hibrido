import os

# Parámetros del Algoritmo (Esto SI se sube)
DIMS = 30
RUNS = 30
EPOCHS = 1000
POP_SIZE = 100
EPOCHS_COMBINATORIA = 3000

# Rutas (Hacerlas relativas para que funcionen en cualquier compu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados') # Todo se guardará en la carpeta 'results'

# Crear la carpeta de resultados si no existe (buena práctica)
os.makedirs(RESULTS_DIR, exist_ok=True)