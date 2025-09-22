import numpy as np

A = np.array([
    [2,1,0,0,0,0],
    [1,2,1,0,0,0],
    [0,1,2,1,0,0],
    [0,0,1,3,1,0],
    [0,0,0,1,3,1],
    [0,0,0,0,1,4],
])

def cholesky_factorization(A):

    Ai = A
    n = len(Ai)
    D = np.zeros((n, n))
    L = np.eye(n, n)
    
    for k in range(len(A)):
        
        lk = Ai[:, k] / Ai[:, k][k]
        outer_product = np.outer(lk, lk)

        Dkk = Ai[k][k]
        Ai = Ai - Dkk * outer_product

        D[k, k] = Dkk
        L[:, k] = lk

    return L, D

L, D = cholesky_factorization(A)

print(np.allclose(A, L @ D @ L.T))