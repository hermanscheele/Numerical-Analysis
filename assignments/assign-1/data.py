import numpy as np
import matplotlib.pyplot as plt


# First dataset
n = 30
start = -2
stop = 2
x = np.linspace(start, stop, n)
eps = 1
np.random.seed(1)
r = np.random.rand(n) * eps
y = x * (np.cos(r + 0.5 * x**3) + np.sin(0.5 * x**3))

# Second dataset
y2 = 4 * x**5 - 5 * x**4 - 20 * x**3 + 10 * x**2 + 40 * x + 10 + r

