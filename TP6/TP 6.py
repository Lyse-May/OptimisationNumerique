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
    plt.legend()
    plt.show()
    
    err = [U[i] - U_exact[i] for i in range(0,N)]
    plt.plot(x,err, label = 'Erreur')
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
    plt.legend()
    plt.show()
    
    for j in range(0,len(N)):
        err.append([U[j][i] - U_exact[i] for i in range(0,len(U[j]))])
    plt.plot(x[0],err[0], label = 'N = 5')
    plt.plot(x[1],err[1], label = 'N = 10')
    plt.plot(x[2],err[2], label = 'N = 15')
    plt.plot(x[3],err[3], label = 'N = 20')
    plt.legend()
    plt.show()
    return 0


comparaison()


"""
PARTIE 2
"""

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
    #f = -np.ones(N**2)
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
    
    """
    NE FONCTIONNE PAS
    """
    
    U,X,Y,V = resolution_Poisson(N, cte)
    ax = plt.gca(projection='3d')
    ax.plot_surface(X,Y,U,cmap=cm.viridis)
    plt.title('Cholesky')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    for l in range(0,100):
        for m in range(0,100):
            Ulm = 16/(((2*l-1)**2 + (2*m-1)**2)* (np.pi**4) * (2*l-1)*(2*m-1))
            U_exact_P = Ulm * np.sin((2*l-1)*np.pi*X) * np.sin((2*m-1)*np.pi*Y)
    
    ax1 = plt.gca(projection='3d')
    ax1.plot_surface(X,Y,U_exact_P,cmap=cm.viridis)
    plt.title('Exact')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()
    return 0

graphPoisson(15, 1)