# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:52:08 2025

@author: ALVARO
"""

import numpy as np

def leer_circuito(archivo):
    """Lee un archivo .cir y extrae sus datos"""
    cir_el, cir_nd, cir_val, cir_ctr = [], [], [], []
    
    with open(archivo, 'r') as f:
        for linea in f:
            partes = linea.split()
            if len(partes) < 5:
                continue  # Ignorar líneas vacías o incompletas
            
            cir_el.append(partes[0])  # Nombre del elemento
            cir_nd.append([int(n) for n in partes[1:5]])  # Nodos
            cir_val.append([float(v) for v in partes[5:8]])  # Valores
            cir_ctr.append(partes[8] if len(partes) > 8 else "0")  # Control
    
    return np.array(cir_el), np.array(cir_nd), np.array(cir_val), np.array(cir_ctr)

def obtener_nodos(cir_nd):
    """Devuelve la lista única de nodos en el circuito"""
    nodos = np.unique(cir_nd)
    return nodos[nodos != 0]  # Excluir nodo de referencia 0

def construir_matriz_incidencia(cir_el, cir_nd):
    """Genera la matriz de incidencia Aa"""
    nodos = obtener_nodos(cir_nd)
    Aa = np.zeros((len(nodos), len(cir_el)))

    for j, (elem, nodos_elem) in enumerate(zip(cir_el, cir_nd)):
        if nodos_elem[0] in nodos:
            Aa[np.where(nodos == nodos_elem[0]), j] = 1
        if nodos_elem[1] in nodos:
            Aa[np.where(nodos == nodos_elem[1]), j] = -1
    
    return Aa

def validar_circuito(cir_nd):
    """Comprueba errores en el circuito"""
    if 0 not in cir_nd:
        print("Error: No hay nodo de referencia (0).")
    else:
        print("Circuito válido.")

# MAIN
archivo = "circuito.cir"  # Nombre del archivo de entrada
cir_el, cir_nd, cir_val, cir_ctr = leer_circuito(archivo)
print("Elementos:", cir_el)
print("Nodos:", obtener_nodos(cir_nd))
Aa = construir_matriz_incidencia(cir_el, cir_nd)
print("Matriz de incidencia:\n", Aa)
validar_circuito(cir_nd)
