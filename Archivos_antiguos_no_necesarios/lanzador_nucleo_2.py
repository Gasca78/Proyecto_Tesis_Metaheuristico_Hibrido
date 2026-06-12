# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:01:38 2026

@author: oswal
"""

import subprocess

benchmarks = ["cec2017", "tsp", "ingenieria"]
modelo = "Markov_80_15_5"

print(f"=== INICIANDO COLA DE TRABAJO - NÚCLEO 2 ({modelo}) ===")

for bench in benchmarks:
    print(f"\n[+] Lanzando {modelo} para el benchmark: {bench}")
    
    comando = ["python", "main_adaptado.py", "--modelo", modelo, "--benchmark", bench]
    resultado = subprocess.run(comando)
    
    if resultado.returncode != 0:
        print(f"[!] Error detectado corriendo {bench}. Pasando al siguiente...")

print("\n=== NÚCLEO 2 FINALIZADO ===")