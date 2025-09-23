import numpy as np

def back_substitution_solver(A, b):
    n = len(A)
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1][n-1]

    for i in range(n - 2, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += A[i][j] * x[j]

        x[i] = (b[i] - s) / A[i][i]

    return x


def QR(A):
    return np.linalg.qr(A)


def Vandermonde(x, m):

    n = len(x)
    A = np.zeros((n,m))
    A[:, 0] = np.ones(n)

    for i in range(1, m):
        ai = x ** i
        A[:, i] = ai

    return A
