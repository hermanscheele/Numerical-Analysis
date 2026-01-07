import numpy as np


def cholesky(A):
    
    n = len(A)
    L = []
    D = []
    Ak = A

    for k in range(n):
        lk = 1 / Ak[k][k] * Ak[k]
        Dkk = Ak[k][k]

        lklkT = lk * lk.T
        Ak = Ak - Dkk * lklkT

        L.append(lk)
        D.append()  
        