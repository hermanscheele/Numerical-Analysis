from data import x, y, y2, n
from methods import back_substitution_solver, QR, Vandermonde
import numpy as np

m = 5
A = Vandermonde(x, m)
print(A)

