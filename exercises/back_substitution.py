import numpy as np

A = np.array([

    [1,2,4],
    [0,9,5],
    [0,0,3]

])

b = np.array([1, 2, 3])

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

x = np.linalg.solve(A, b)
x2 = back_substitution_solver(A, b)    

print(np.all(x == x2))
