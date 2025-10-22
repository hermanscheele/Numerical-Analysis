import numpy as np
import matplotlib.pyplot as plt

# Compute the SVD(A) = U S V_T
def svd(a):
    n = a.shape[0]
    m = a.shape[1]

    # Comput the normal matrix
    norm_mat = a.T @ a
    
    # Compute the eignevalue pairs
    eigpairs = np.linalg.eigh(norm_mat)
    eigvals = eigpairs[0]
    eigvecs = eigpairs[1]
    
    # Reorder the eigpairs
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Force small negative values to 0
    eigvals = np.clip(eigvals, 0, None)  # Replace negatives with 0
    
    # Compute sinular values and matrix s
    sing_vals = np.sqrt(eigvals)  

    # Compute U, S & V
    s = np.diag(sing_vals)
    u = (a @ eigvecs) / sing_vals
    v = eigvecs
    
    return u, s, v.T


# Plot the log10 of the singular values a image list
def plot_log_singular_vals(ims):

    for name, im in ims:
        u, s, v_T = svd(im)
        sigmas = np.diag(s)
                
        log_sig = np.log10(sigmas)
        x = np.linspace(1, len(log_sig), len(log_sig))
        plt.plot(x, log_sig, label=name)

    plt.xlabel('$\sigma_i$')
    plt.ylabel('$\log_{10}(\sigma)$')
    plt.legend()
    plt.show()
    

# Plot the compressed image with r
def plot_compressed(im, r):
    
    U, S, V_T = svd(im)
    sigmas = np.diag(S)[:r]

    u = U[:, :r]
    s = np.diag(sigmas)
    v = V_T.T[:, :r]

    im_c = u @ s @ v.T
    
    plt.figure()
    plt.title(f'r = {r}')
    plt.axis(False)
    plt.imshow(im_c, cmap='gray')
    plt.show()


