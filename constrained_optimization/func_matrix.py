import numpy as np
import matplotlib as mt


def weighted_sphere_function(x):
    weighted_value = 0
    for j, k in enumerate(x):
        weighted_value = weighted_value + ((j + 1) * k ** 2)
    return weighted_value

def w_sphere_matrix(x,dim):
    natural_numbers = np.arange(1, dim + 1)
    wv = (x*natural_numbers) @ np.transpose(x)
    return wv

def exact_gradient(x):
    i = x.shape
    grad_vector = np.zeros(i)
    for j, k in enumerate(x):
        grad_vector[j] = (2 * (j + 1) * k)
    return grad_vector


def grad_matrix(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    gm = 2 * x * natural_numbers
    return gm

def approx_gradient(x, l, wv):
    h = 10 ** -l * np.linalg.norm(x)
    i = x.shape[0]
    fin_diff = np.zeros(i)
    for j, k in enumerate(x):
        a = np.array(x[0:j])
        b = np.array(x[j+1:])
        xhp = np.concatenate((a, x[j]+h, b), axis=None)
        xhn = np.concatenate((a, x[j]-h, b), axis=None)
        fin_diff[j] = (weighted_sphere_function(xhp, i) -
                       weighted_sphere_function(xhn, i)) / h
    return fin_diff


def approx_gradient_centered(x, l, wv):
    h = 10 ** -l * np.linalg.norm(x)
    i = x.shape[0]
    fin_diff = np.zeros(i)
    #ei = np.zeros(i)
    for j, k in enumerate(x):
        xh = np.concatenate((x[0:j], x[j]+h, x[j+1:]), axis=None)
        fin_diff[j] = (weighted_sphere_function(xh, i) - wv) / h
    return fin_diff

def Pi_x(x):
    x_projected = x
    x_projected[np.where(x < -5.12)] = -5.12
    x_projected[np.where(x > 5.12)] = 5.12
    return x_projected


def f_armijo_condition(fx, gradfx, c1, alpha, direcx):
    fk_arm = fx + c1 * alpha * (gradfx @ direcx)
    return fk_arm


if __name__ == '__main__':

    n_val = 10 ** 2

    # stablishing x_0 in  n_val
    x0 = np.random.default_rng(seed=42).uniform(-6, 6, n_val)
    print(x0)

    # DEFINING THE VECTOR DE PRUEBA
     # np.array([154, -220, 40])
    # vector_center = np.zeros(s_arr)
    # vector_boundaries = np.repeat(limit_values, s_arr, axis=1)
    # print(vector_boundaries)

    # VARIABLES NEEDED FOR THE PROCESS
    # number max of iterations
    kmax = 2000
    btmax = 100
    tolerance = 1e-12
    gamma = 1e-1
    c1 = 1e-4
    rho = 0.8

    # Defining the dominio
    limit_values = np.array([-5.12, 5.12])

    # INIZIALIZATIONS FOR THE PROCESS:

    k = 0  # number of iterations
    # checking if the x0 is out of bounds
    xk = x0
    xk = Pi_x(xk)

    # defining the values of the function for x_PRUEBA
    f_xk = weighted_sphere_function(xk)

    f_new = w_sphere_matrix(xk, n_val)
    # definig the exact gradient:
    g_xk = exact_gradient(xk)

    g_new = grad_matrix(xk, n_val)

    gradfk_norm = np.linalg.norm(g_xk)
    deltaxk_norm = tolerance + 1

    print(f"X projected on the dominium: {xk}")
    print(f"Funcion con for: {f_xk}")
    print(f"Funcion con array: {f_new}")
    print(f"Gradiente con for: {g_xk}")
    print(f"Gradiente con array: {g_new}")

    print(f"Number of iterations: {k}, norm gradient: {gradfk_norm}, value function f(x)= {f_xk}")
    # print(f"Matrix of x:{xseq}")

