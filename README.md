# Algoritmo Híbrido JADE-PSO-GA para Optimización Global

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Research-orange)

Este repositorio contiene el código fuente y los experimentos de mi Tesis de Maestría. El proyecto implementa una **metaheurística híbrida** que combina las fortalezas de tres algoritmos clásicos para resolver problemas de optimización numérica de alta dimensionalidad.

## 🚀 Descripción del Algoritmo

La propuesta integra tres estrategias de búsqueda que se seleccionan dinámicamente durante la evolución:

1.  **JADE (Adaptive Differential Evolution):** Para explotación y refinamiento, utilizando adaptación de parámetros ($\mu CR, \mu F$).
2.  **PSO (Particle Swarm Optimization):** Para mejorar la velocidad de convergencia y exploración global.
3.  **GA (Genetic Algorithms):** Operadores de cruce y mutación para mantener la diversidad genética.

El algoritmo incluye un mecanismo de **Probabilidades Dinámicas** (Roulette Wheel Selection) y **Cadenas de Markov** (en desarrollo) para adaptar la estrategia de búsqueda según el éxito en las iteraciones recientes.

## 🧪 Benchmarks Utilizados

El rendimiento se evalúa utilizando las suites de funciones estándar de la computación evolutiva:

* **CEC 2017:** Funciones F1 a F29 (Unimodales, Multimodales, Híbridas y Composición).
* **CEC 2020:** Validación adicional.
* **Dimensiones:** Pruebas de escalabilidad en 30, 50 y 100 dimensiones.

## 📋 Estructura del Proyecto

* **HIBRIDO.py**: Clase principal con la lógica del algoritmo (hereda de `Mealpy.Optimizer`).
* **main.py**: Script orquestador que ejecuta los experimentos, calcula estadísticas y genera reportes.
* **config.py**: Archivo de configuración global (Número de Corridas, Épocas, Dimensiones, Rutas).
* **requirements.txt**: Lista de dependencias y librerías necesarias.
* **.gitignore**: Archivo para excluir resultados pesados y temporales del control de versiones.
* **results/**: Carpeta generada automáticamente (No se sube al repositorio). Contiene:
    * **Fitness.csv**: Valores finales de optimización.
    * **Tiempos.csv**: Costo computacional por corrida.
    * **Convergencia.csv**: Historial promedio por época.
    * **Diversidad.csv**: Métricas de exploración/explotación dimensional.

## 🛠️ Instalación y Requisitos

Este proyecto utiliza `Miniforge` (o Anaconda) con **Python 3.10**.

1. **Clonar el repositorio:**

        git clone https://github.com/TuUsuario/TuRepositorio.git
        cd TuRepositorio

2. **Crear el entorno virtual (Recomendado):**

        mamba create -n TuEntorno python=3.10
        mamba activate TuEntorno

3. **Instalar dependencias:**

        # Librerías base
        mamba install numpy pandas scipy matplotlib seaborn scikit-learn
   
        # Librerías de optimización (vía pip)
        pip install mealpy opfunu

## 📊 Ejecución

Para correr el benchmark configurado en `config.py`:

    python main.py

El script detectará automáticamente la configuración de hardware y ejecutará las corridas, guardando los resultados organizados por fecha en la carpeta `resultados/`.

## 📈 Resultados Destacados (30 Dimensiones) (Preliminares)

La implementación de probabilidades dinámicas ha logrado una mejora drástica en la eficiencia computacional:

* **Velocidad:** Speedup promedio de **2.30x** (54% de ahorro de tiempo) comparado con JADE.
* **Eficiencia:** En funciones unimodales (F1), logra la misma calidad de solución en **menos de la mitad del tiempo**.
* **Robustez:** Mantiene su superioridad en funciones multimodal engañosas (F9 Schwefel).

## ✒️ Autor

* **Oswaldo Gasca** - *Desarrollo e Investigación* - oswaldo.gasca9379@alumnos.udg.mx | [LinkedIn](https://www.linkedin.com/in/oswaldo-alejandro-gasca-ramos-2705vb/)

---
*Proyecto desarrollado como parte de la investigación de Maestría en Universidad de Guadalajara.*
