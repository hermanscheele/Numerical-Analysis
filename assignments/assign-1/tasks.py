from data import x, y, y2, n
from methods import back_sub_solver, QR, Vandermonde, forward_sub_solver, cholesky_fact
import numpy as np
import matplotlib.pyplot as plt 


ms = [3, 8]
for m in ms: 
    data = [y, y2]
    for yi in data:


        # ------------ Task 1 ------------ #

        A = Vandermonde(x, m)
        b = yi

        Q, R = QR(A)
        c = Q.T @ b

        R1 = R[:m, :m]
        c1 = c[:m]

        coeff = back_sub_solver(R1, c1)


        # ------------ Task 2 ------------ #

        B = A.T @ A
        e = A.T @ b

        L, D = cholesky_fact(B)
        R = L @ np.sqrt(D)

        u = back_sub_solver(R, e)
        coeff2 = forward_sub_solver(R.T, u)


        # ------------ Plot ------------ #

        # Plot of fitted pol. by QR-fact.
        xplot = np.linspace(x.min(), x.max(), 100)
        Aplot = Vandermonde(xplot, m)
        yfit = Aplot @ coeff
        plt.plot(x, yi, 'o', label="data")
        plt.plot(xplot, yfit, label=f"ŷ(x) (m = {m})")
        plt.xlabel("x-values")
        plt.ylabel("ŷ(x)")
        plt.legend()
        plt.show()

        # Plot of fitted pol. by Normal eqs.
        # xplot2 = np.linspace(x.min(), x.max(), 100)
        # Aplot2 = Vandermonde(xplot2, m)
        # yfit2 = Aplot2 @ coeff
        # plt.plot(x, y, 'o')
        # plt.plot(xplot2, yfit2)
        # plt.show()