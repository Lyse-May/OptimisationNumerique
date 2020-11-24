# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:20:42 2020

@author: Capucine
"""


import numpy as np
from matplotlib import pyplot as plt
import math
from math import exp

"Partie 1 "

#Définition de la matrice A

f = lambda x : exp(x)
N = 100
h = 1/N
c = np.zeros((N,1))
c[0]= 1/(h**2)
c[-1] = (math.e)/(h**2)
x = np.linspace(0.,1.,N)

v = np.zeros(N)

for i in range (0,N):
    v[i] = f(x[i]) + c[i]
    

def matrice(n):
    # Création de la matrice A
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = 2*(1+(h**2))
       
    for i in range(1,n):
        A[i,i-1] = -1.
        A[i-1,i] = -1.
       
    return (1/h**2)*A

def trianginf (T,b):
   # T est une matrice triangulaire inférieure
   # b est le vecteur second membre
   n = len(b)
   x = np.zeros((n,1))
   x[0] =  b[0]/T[0,0]
   for i in range(1,n):
       s = 0
       for j in range(0,i):
           s += T[i,j] * x[j]
       x[i] = (1/T[i,i]) * (b[i] - s)
   return x


def triansup(T,b):
   # T est une matrice triangulaire supérieure
   # b est le vecteur second membre
   n = len(b)
   x = np.zeros((n,1))
   x[n-1] =  b[n-1]/T[n-1,n-1]
   for i in range(n-2,-1,-1):
       s = 0
       for j in range(i+1,n):
           s += T[i,j] * x[j]
       x[i] = (1/T[i,i]) * (b[i] - s)
   return x

def Cholesky(A):
    # Décomposition de la matrice A par la méthode de Cholesky
    n = A.shape[0]
    C = np.zeros((n,n))
   
    C[0,0] = np.sqrt(A[0,0])
   
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

def ludecomp(A):
    # A est la matrice principale
    n = A.shape[0]
    U = np.zeros((n,n))
    L = np.zeros((n,n))
   
    for j in range(0,n):
        U[0,j] = A[0,j]
       
    for i in range(1,n):
        L[i,0] = A[i,0]/A[0,0]
       
    for i in range(1,n):
        L[i,i] = 1.
        s1 = 0.
        for k in range(0,i):
            s1 += L[i,k] * U[k,i]
        U[i,i] = A[i,i] - s1
       
        for j in range(i+1, n):
            s2 = 0.
           
            for k in range(0,i):
                s2 += L[i,k] * U[k,j]
            U[i,j] = A[i,j] - s2
           
            s3 = 0.
            for k in range(0,i):
                s3 += L[j,k] * U[k,i]
            L[j,i] = (1/U[i,i]) * (A[j,i] - s3)
   
    L[0,0] = 1       
       
    return [L,U]

def lusolve(A,b):
    # A est la matrice principale
    # b est le vecteur second membre
    [L,U] = ludecomp(A)
    y = trianginf(L,b)
    x = triansup(U,y)
    return x

u_LU = lusolve(matrice(N),v)
u_Cholesky = resolchol(matrice(N),v)
#print(u_Cholesky)

#Solution exacte
u_reel = [ f(x[i]) for i in range(0,N) ]

plt.plot(x,u_reel,label = "u_reel")
plt.plot(x,u_LU,label ="u_LU")
plt.plot(x,u_Cholesky,label ="u_Cholesky")
plt.xlabel('x')
plt.ylabel('u')
plt.title("Solution approchée et solution exacte")
plt.legend()
plt.show()

"La solution approchée et la solution exacte se superposent parfaitement quand N augmente"

"Erreur"

err_Cholesky = np.zeros(N)
err_LU = np.zeros(N)

for i in range (0,N):
        
        err_LU[i] = abs(u_LU[i] - u_reel[i])
        err_Cholesky[i] = abs(u_Cholesky[i] - u_reel[i])
        

plt.plot(x, err_Cholesky, label = "Erreur avec Cholesky")
plt.plot(x, err_LU, label = "Erreur avec LU")
plt.xlabel('x')
plt.ylabel('Erreur')
plt.title("-u'' +  2u = f avec f = exp(x)")
plt.legend()
plt.show()