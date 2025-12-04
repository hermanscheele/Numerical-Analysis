

def newtons_meth(a, n):

    def f(x):
        return x**2 - a

    def f_d(x):
        return 2*x
    
    x_k = 1.0
    print(x_k)

    for k in range(n):
        x_k = x_k - (f(x_k) / f_d(x_k))
        print(x_k)

    return x_k


a = 81
n = 10

newtons_meth(a, n)

