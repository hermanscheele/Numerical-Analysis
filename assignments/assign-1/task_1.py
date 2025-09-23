from data import x, y, y2, n
from methods import back_substitution_solver, QR, Vandermonde
import numpy as np
import matplotlib.pyplot as plt 

m = 8
A = Vandermonde(x, m)
b = y

Q, R = QR(A)
c = Q.T @ b

R1 = R[:m, :m]
c1 = c[:m]

coeff = back_substitution_solver(R1, c1)

xplot = np.linspace(x.min(), x.max(), 100)
Aplot = Vandermonde(xplot, m)
yfit = Aplot @ coeff

plt.plot(x, y, 'o')
plt.plot(xplot, yfit)
plt.show()
