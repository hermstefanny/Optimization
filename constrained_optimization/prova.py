import numpy as np
import matplotlib as mt


def weighted_sphere_function(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    wv = (x * natural_numbers) @ np.transpose(x)
    return wv


def approx_gradient(x, l, wv):
    h = 10 ** -l * np.linalg.norm(x)
    #print(f"The value of h {h}")
    i = x.shape[0]
    fin_diff = np.zeros(i)
    ei = np.zeros(i)
    for j, k in enumerate(x):
        ei[j] = h
        fin_diff[j] = (weighted_sphere_function(x+ei, i) - wv) / h
        ei[j] = 0
    return fin_diff


def exact_gradient(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    grad_vector = 2 * x * natural_numbers
    return grad_vector


if __name__ == '__main__':
    n_val = 10**3

    l = 6

    # establishing x_0 in  n_val
    x0 = np.random.default_rng(seed=42).uniform(-5.12, 5.12, n_val)
    #print(x0)

    f_xk = weighted_sphere_function(x0, n_val)
    # defining the exact gradient:
    g_xk = exact_gradient(x0, n_val)

    gradfk_norm = np.linalg.norm(g_xk)

    ag_xk = approx_gradient(x0, l, f_xk)
    grad_norm_fin = np.linalg.norm(ag_xk)
    diff = abs(g_xk) - abs(ag_xk)
    print(f"\nEl valor de la funcion en el punto es: {f_xk}, "
          f"\nEl valor de la norma del gradiente exacto es {gradfk_norm}"
          f"\nEl valor de la norma del gradiente aproximado es {grad_norm_fin}")
