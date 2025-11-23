import numpy as np


def backsolve(A, b):
    
    n = len(A) - 1
    x = [0] * (n + 1)
    print(b)
    x[n] = b[n] / A[n][n]   # 1 flop

    print(" ")
    print(x)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - (A[i][i + 1] * x[i + 1])) / A[i][i]   # 3(n - 1) flops
        print(x)

    return x





# --------- Test --------- #

A = [[1,2,3], [0,5,6], [0,0,9]]  # upper triangular
b = [2, 4, 8]

x = np.linalg.solve(A, b)
xback = backsolve(A, b)

print(x)
print(xback)
print(np.all(x == xback))



# -------- Dev -------- #

# A = np.array([

#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
    
# ])

# A = [[1,2,3], [4,5,6], [7,8,9]]


# n_idx = len(A) - 1
# a_nn = A[n_idx][n_idx]
# print(a_nn)


# x = [0] * 5
# x[1] = 33
# print(x)


# print("")
# n = len(A) - 1
# for i in range(n-1, -1, -1):
#     print(i)