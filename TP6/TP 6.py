#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:22:35 2020

@author: HOAREAU.LyseMay
"""

import time
import numpy as np
from matplotlib import cm
from Cholesky import resolchol
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

"""
PARTIE 1
"""
print("Partie 1")

f = lambda x : np.exp(x)
u_exact = lambda x : np.exp(x)

u_0 = 1
u_1 = np.exp(1)

def matrice(n,h):
    # Création des matrices A
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = 2*(1+h**2)
        
    for i in range(1,n):
        A[i,i-1] = -1.
        A[i-1,i] = -1.
        
    return (1/h**2)*A

def resolution(N,u_0,u_1,f):
    h = 1/N
    A = matrice(N,h)
    x = np.linspace(0,1,N)
    b = f(x)
    b[0] += u_0/(h**2)
    b[-1] += u_1/(h**2)
    U = resolchol(A,b)
    
    return U,x

def graph(N):
    U,x = resolution(N, u_0, u_1,f)
    U_exact = u_exact(x)
    
    plt.plot(x,U, label = "Cholesky")
    plt.plot(x,U_exact, label = "Exacte")
    plt.title("Solution de Cholesky et la solution exacte")
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()
    
    err = [abs(U[i] - U_exact[i]) for i in range(0,N)]
    plt.plot(x,err, label = 'Erreur')
    plt.title("Erreur entre la solution de Cholesky et la solution exacte")
    plt.xlabel('x')
    plt.ylabel('Erreur')
    plt.legend()
    plt.show()
    
    return 0

graph(N = 20)

def comparaison():
     
    U = []
    N = np.linspace(5,20,4)
    x = []
    err = []
    
    for i in range(0,len(N)):
        U.append(resolution(int(N[i]),u_0,u_1,f)[0])
        x.append(resolution(int(N[i]),u_0,u_1,f)[1])
        
    U_exact = u_exact(x[3])
    plt.plot(x[0],U[0], label = "N = 5")
    plt.plot(x[1],U[1], label = "N = 10")
    plt.plot(x[2],U[2], label = "N = 15")
    plt.plot(x[3],U[3], label = "N = 20")
    plt.plot(x[3],U_exact, label = "Exacte", color = 'r')
    plt.title("Comparaison entre la solution de Cholesky et la solution exacte avec différentes tailles de N")
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.show()
    
    for j in range(0,len(N)):
        err.append([abs(U[j][i] - U_exact[i]) for i in range(0,len(U[j]))])
    plt.plot(x[0],err[0], label = 'N = 5')
    plt.plot(x[1],err[1], label = 'N = 10')
    plt.plot(x[2],err[2], label = 'N = 15')
    plt.plot(x[3],err[3], label = 'N = 20')
    plt.title("Comparaison des erreurs en fonction des différentes tailles de N")
    plt.xlabel('x')
    plt.ylabel('Erreur')
    plt.legend()
    plt.show()
    return 0

"La solution approchée  et la solution exacte se superposent parfaitement quand N augmente"


comparaison()


"""
PARTIE 2
"""
print("Partie 2")

def matrice_Poisson(n,h):
    # Création des matrices A
    N = n**2
    A = np.zeros((N,N))
    for i in range(0,N):
        A[i,i] = -4.
        
    for i in range(1,N):
        A[i,i-1] = 1.
        A[i-1,i] = 1.
        if i%n == 0:
            A[i,i-1] = 0.
            A[i-1,i] = 0.
    A += np.eye(N,N,n)
    A += np.eye(N,N,-n)
    return -(1/h**2)*A

def resolution_Poisson(N,cte):
    h = 1/N
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,y)
    
    A = matrice_Poisson(N,h)
    b = cte * np.ones(N**2)
    V = resolchol(A,b)
    U = np.zeros((N,N))
    for i in range(1,N-1):
        for j in range(1,N-1):
            k = (i-1)*N+j
            U[i,j] = V[k]
            
    return U,X,Y,V

def graphPoisson(N,cte):
        
    U,X,Y,V = resolution_Poisson(N, cte)
    ax = plt.gca(projection='3d')
    #ax.plot_surface(X,Y,U,cmap=cm.viridis)
    ax.plot_surface(X,Y,U,cmap='jet')
    plt.title('Cholesky')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    return 0

def graphPoissonExact(N):

    U_exact_P = 0
    p = 0
    
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,y)
    
    for l in range(1,500):
        for m in range(1,500):
            Ulm = -16/(((2*l-1)**2 + (2*m-1)**2)* (np.pi**4) * (2*l-1)*(2*m-1))
            U_exact_P += Ulm * np.sin((2*l-1)*np.pi*X) * np.sin((2*m-1)*np.pi*Y)
            
        p += U_exact_P
         
    ax = plt.gca(projection='3d')
    #ax.plot_surface(X,Y,U_exact_P,cmap=cm.viridis)
    ax.plot_surface(X,Y,U_exact_P,cmap='jet')
    plt.title('Exact')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    return 0
    
print("Question 1 avec f = 1")
graphPoisson(15, 1) #Question 1
print("Question 2 avec f = -1")
graphPoisson(15, -1) #Question 2
print("Solution exacte")
graphPoissonExact(15)

"""
PARTIE 3
"""
print("Partie 3")

L = 1.0
tfin = 0.2

dx = 1.0e-2
dt = 1.0e-4

Nx = int(L/dx)
Nt = int(tfin/dt)

eta = dt / ((dx)**2)

X = np.linspace(0,1,Nx)
T = np.linspace(0,tfin,Nt)
SX,ST = np.meshgrid(X,T)

def matrice_chaleur1D(n,h):
    # Création des matrices A
    A = np.zeros((n,n),float)
    A[0,0] = A[n-1,n-1] = 1
    for i in range(0,n): # ou for i in range(1,n-1)
        A[i,i] = (1+2*h)
        
    for i in range(1,n):
        A[i,i-1] = h
        A[i-1,i] = h
        
    return A

A = matrice_chaleur1D(Nx,eta)
V = np.zeros((Nt,Nx),float)

V[0,:] = np.exp(X)
#V[0,int(Nx/2)] = 1/2


for t in range(0,Nt-1):
    V[t+1,:] = A.dot(V[t,:].T)
    #V[t+1,:] = np.linalg.solve(A,V[t,:]).T
    
fig = plt.figure(figsize=(14,8))
ax = plt.gca(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('temps')
ax.set_zlabel('temperature')
ax.view_init(elev=15, azim = 120)
ax.plot_surface(SX,ST,V,cstride=1,linewidth=0,cmap='jet')

