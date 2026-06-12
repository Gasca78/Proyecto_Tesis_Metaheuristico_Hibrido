# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:00:16 2026

@author: oswal
"""
import subprocess

benchmarks = ["cec2017", "tsp", "ingenieria"]
modelo = "Markov_WTA"

print(f"=== INICIANDO COLA DE TRABAJO - NÚCLEO 1 ({modelo}) ===")

for bench in benchmarks:
    print(f"\n[+] Lanzando {modelo} para el benchmark: {bench}")
    
    # Llamada a tu main.py pasándole los argumentos
    comando = ["python", "main_adaptado.py", "--modelo", modelo, "--benchmark", bench]
    resultado = subprocess.run(comando)
    
    if resultado.returncode != 0:
        print(f"[!] Error detectado corriendo {bench}. Pasando al siguiente...")

print("\n=== NÚCLEO 1 FINALIZADO ===")