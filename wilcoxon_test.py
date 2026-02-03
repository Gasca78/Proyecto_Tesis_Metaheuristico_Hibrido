# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:32:36 2025

@author: oswal
"""

# Prueba de Wilcoxon

# Importamos librerías
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

# Carga de datos 
df_hibrido = pd.read_csv('Resultados_hibrid_JADE_CEC2020.csv', index_col=0)
df_JADE = pd.read_csv('Resultados_JADE_CEC2020.csv', index_col=0)

################ WILCOXON TEST ################

def evaluacion_wilcoxon(df_hibrido, df_JADE):
  # Hacemos la prueba por cada columna
  alpha = 0.05
  epsilon = 1e-8 # Tolerancia para diferencias muy pequeñas
  # Definimos unos arreglos para almacenar el listado de ganados por cada modelo
  ganador_hibrido = []
  ganador_jade = []
  empates = []
  dif_significativas = []
  # Obtenemos los nombres cortos de las funciones
  nc = df_hibrido.columns.str.extract(r'(F\d+)', expand=False)
  # Diccionario para relacionar los nombres largos con los cortos
  nombres_cortos = dict(zip(df_hibrido.columns, nc))

  for func in df_hibrido.columns:
    # Tenemos las 2 columnas de datos
    data_h = df_hibrido[func]
    data_j = df_JADE[func]

    print(func)
    # Evaluamos primero para descartar diferencias muy pequeñas
    diff_abs = np.mean(np.abs(data_h - data_j))
    if diff_abs < epsilon:
      ganador = "Empate"
      stats = 0
      pvalue = 1.0
      print(f"Stats: {stats}, pvalue: {pvalue}, ganador: {ganador}\n")
      empates.append(nombres_cortos[func])
    else:
      # Hacemos la prueba de Wilcoxon
      stats, pvalue = wilcoxon(df_hibrido[func], df_JADE[func])

      if pvalue < alpha: # Si hay diferencia significativa
        mean_h = data_h.mean()
        mean_j = data_j.mean()
        if mean_h < mean_j:
          ganador = "Híbrido"
          print(f"Stats: {stats}, pvalue: {pvalue}, ganador: {ganador}\n")
          ganador_hibrido.append(nombres_cortos[func])
        else:
          ganador = "JADE"
          print(f"Stats: {stats}, pvalue: {pvalue}, ganador: {ganador}\n")
          ganador_jade.append(nombres_cortos[func])
      else:
        print(f"Stats: {stats}, pvalue: {pvalue}, ganador: No hay diferencia significativa\n")
        dif_significativas.append(nombres_cortos[func])



  print(f"Ganadores Híbrido: {ganador_hibrido}")
  print(f"Ganadores JADE: {ganador_jade}")
  print(f"Total de ganadores Híbrido: {len(ganador_hibrido)}")
  print(f"Total de ganadores JADE: {len(ganador_jade)}")
  print(f"Empates: {empates}")
  print(f"Total de empates: {len(empates)}")
  print(f"Diferencias no significativas: {dif_significativas}")
  print(f"Total de diferencias no significativas: {len(dif_significativas)}")
  return ganador_hibrido

################ EVALUACIÓN TIEMPO ################

def evaluacion_tiempos(df_tiempo_hibrido, df_tiempo_JADE):
    alpha = 0.05
    # Listas para construir el DataFrame final
    data_resumen = []

    # Obtenemos los nombres cortos de las funciones
    nc = df_tiempo_hibrido.columns.str.extract(r'(F\d+)', expand=False)
    # Diccionario para relacionar los nombres largos con los cortos
    nombres_cortos = dict(zip(df_tiempo_hibrido.columns, nc))

    print(f"{'FUNCIÓN':<10} | {'T. MEDIA HÍBRIDO (s)':<20} | {'T. MEDIA JADE (s)':<20} | {'SPEEDUP':<10} | {'MEJORA %':<10} | {'WILCOXON'}")
    print("-" * 105)

    conteo_mas_rapido = 0

    for func in df_tiempo_hibrido.columns:
      # Tenemos las 2 columnas de datos
      data_h = df_tiempo_hibrido[func]
      data_j = df_tiempo_JADE[func]
      # Obtenemos las medias
      mean_h = data_h.mean()
      mean_j = data_j.mean()

      # Calculamos el Speed Up
      speed_up = mean_j / mean_h if mean_h != 0 else 0
      # Calculamos la mejora porcentual
      mejora_perc = ((mean_j-mean_h)/mean_j)*100

      try:
        # Hacemos la prueba de Wilcoxon
        stats, pvalue = wilcoxon(data_h, data_j)
        sig = "Sí" if pvalue < 0.05 else "No"
      except ValueError:
        sig = "Iguales" # Tiempos idénticos

      # Evaluamos quien ganó:
      if pvalue < alpha: # Si hay diferencia significativa
        if mean_h < mean_j:
          ganador = "Híbrido"
          conteo_mas_rapido += 1
        else:
          ganador = "JADE"
      else:
        ganador = "No hay diferencia significativa"

      # Imprimir fila formateada
      print(f"{nombres_cortos[func]:<10} | {mean_h:^20.4f} | {mean_j:^20.4f} | {speed_up:^10.2f}x | {mejora_perc:^9.2f}% | {sig}")

      # Guardar para análisis posterior
      data_resumen.append({
          'Funcion': nombres_cortos[func],
          'Hibrido_Mean': mean_h,
          'JADE_Mean': mean_j,
          'Speedup': speed_up,
          'Mejora_Pct': mejora_perc,
          'Mas_Rapido': ganador,
          'Significativo': sig
      })

    # Resumen Global
    df_resumen = pd.DataFrame(data_resumen)
    promedio_speedup = df_resumen['Speedup'].mean()
    promedio_ahorro = df_resumen['Mejora_Pct'].mean()

    print("-" * 105)
    print(f"RESUMEN GLOBAL:")
    print(f"  - El Híbrido fue más rápido en {conteo_mas_rapido} de {len(df_tiempo_hibrido.columns)} funciones.")
    print(f"  - Speedup Promedio: {promedio_speedup:.2f}x (Veces más rápido)")
    print(f"  - Ahorro de Tiempo Promedio: {promedio_ahorro:.2f}%")

    return df_resumen

################ GRÁFICAS CONVERGENCIA ################

def grafica_convergencia(df_hibrido, df_jade, ganador_hibrido):
  # Obtenemos los nombres cortos de las funciones
  nc = df_hibrido.columns.str.extract(r'(F\d+)', expand=False)
  # Diccionario para relacionar los nombres largos con los cortos
  nombres_cortos = dict(zip(df_hibrido.columns, nc))
  # Mapeamos los nombres para cambiarlo en los dataframe
  df_hibrido = df_hibrido.rename(columns=nombres_cortos)
  df_jade = df_jade.rename(columns=nombres_cortos)
  # Graficamos la convergencia en las funciones donde el Híbrido le gana a JADE
  df_conv_h_ganados = df_hibrido[ganador_hibrido]
  df_conv_j_perdidos = df_jade[ganador_hibrido]
  # Tomamos las épocas como el eje x y los valores de las columnas como y
  x_h = df_conv_h_ganados.index
  y_h = df_conv_h_ganados.values
  x_j = df_conv_j_perdidos.index
  y_j = df_conv_j_perdidos.values
  # Hacemos un plot para cada formula, mostramos los diferentes comportamientos de cada función
  for i in range(len(df_conv_h_ganados.columns)):
    plt.plot(x_h, y_h[:, i], label=f'{df_conv_h_ganados.columns[i]} Hibrido')
    plt.plot(x_j, y_j[:, i], label=f'{df_conv_h_ganados.columns[i]} JADE')
    plt.xlabel('Épocas')
    plt.ylabel('Fitness')
    plt.title('Convergencia de las Funciones Ganadoras')
    plt.legend()
    plt.show()