import numpy as np



def svd(a):
    n = a.shape[0]
    m = a.shape[1]

    # Comput the normal matrix
    norm_mat = a.T @ a
    
    # Compute the eignevalue pairs
    eigpairs = np.linalg.eig(norm_mat)
    eigvals = eigpairs[0]
    eigvecs = eigpairs[1]
    
    # Reorder the eigpairs
    indicies = np.argsort(eigvals)[::-1]
    eigvals = eigvals[indicies]
    eigvecs = eigvecs[:, indicies]

    # Compute sinular values and matrix s
    sing_vals = np.sqrt(eigvals)
    s = np.diag(sing_vals)
    
    # Compute U and V
    u =  (a @ eigvecs) / sing_vals
    v = eigvecs
    
    return [ u, s, v.T ]







a = np.array([

    [-1, 0, 0],
    [0, 4, 0],
    [0, 0, 3],

])

b = np.array([

    [-1, 0],
    [0, 4],
    [0, 0],

])
print(svd(b)[2])