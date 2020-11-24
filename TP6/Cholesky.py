#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:49:32 2020

@author: HOAREAU.LyseMay
"""
import numpy as np

def triansup(T,b):
   n = len(b)
   x = np.zeros((n,1))
   x[n-1] =  b[n-1]/T[n-1,n-1]
   for i in range(n-2,-1,-1):
       s = 0
       for j in range(i+1,n):
           s += T[i,j] * x[j] 
       x[i] = (1/T[i,i]) * (b[i] - s) 
   return x

def trianginf (T,b):
   n = len(b)
   x = np.zeros((n,1))
   x[0] =  b[0]/T[0,0]
   for i in range(1,n):
       s = 0
       for j in range(0,i):
           s += T[i,j] * x[j] 
       x[i] = (1/T[i,i]) * (b[i] - s) 
   return x

def Cholesky(A):
    # Décomposition de la matrice A par la méthode de Cholesky
    n = A.shape[0]
    C = np.zeros((n,n))
    
    C[0,0] = np.sqrt(np.abs(A[0,0]))
    
    for i in range(1,n):
        C[i,0] = A[i,0]/C[0,0]
        
    for j in range(1,n):
        s1 = 0 
        for k in range(0,j):
            s1 += C[j,k]**2
        C[j,j] = np.sqrt(A[j,j] - s1)
         
        for i in range(j+1, n):
            s2 = 0
            for k in range(0,j):
                s2 += C[i,k]*C[j,k]
            C[i,j] = (1/ C[j,j]) * (A[i,j] - s2)
        
    return C

def resolchol(A,b):
    # Résolution de l'équation AX = b
    C = Cholesky(A)
    y = trianginf(C,b)
    x = triansup(C.T,y)
    return x

#A = np.array([[15., 10, 18, 12],
#             [10, 15, 7, 13],
#             [18, 7, 27, 7],
#             [12, 13, 7, 22]])
#    
#b = np.array([53.,72,26,97])
#
#print(resolchol(A,b))