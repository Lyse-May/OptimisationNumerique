# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:16:49 2020

@author: Capucine
"""

import time
import numpy as np
from matplotlib import pyplot as plt


"Question 1"

#Définition de la matrice A


N = 20 
h = 1/(N+1)
delta_x = h
delta_t = 0.0020
alpha = delta_t/(delta_x**2)
x = np.linspace(0.,1.,N)
t = np.linspace(0.,1.,N)
U = np.zeros((N,N))

for i in range (0,N):
    U[i][0] = 0

def matrice(n):
    # Création de la matrice A
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = 1-2*alpha
       
    for i in range(1,n):
        A[i,i-1] = alpha
        A[i-1,i] = alpha
       
    return A

A = matrice(N)

for i in range (1,N):
    for j in range (1,N):
        U[i][j] = A[i][j] *U[i-1][j-1]
"Pas fini"

