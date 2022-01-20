
import numpy as np
import matplotlib as mt


def weighted_sphere_function(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    wv = (x * natural_numbers) @ np.transpose(x)
    return wv


def approx_gradient(x, lm, wv):
    h = 10**-lm * np.linalg.norm(x)
    i = x.shape[0]
    fin_diff = np.zeros(i)
    ei = np.zeros(i)
    for j, k in enumerate(x):
        ei[j] = h
        fin_diff[j] = (weighted_sphere_function(x+ei, i) - wv)/h
        ei[j] = 0
    return fin_diff


def Pi_x(x):
    x_projected = x
    x_projected[np.where(x < -5.12)] = -5.12
    x_projected[np.where(x > 5.12)] = 5.12
    return x_projected


def f_armijo_condition(fx,  gradfx, c1, alpha, direcx):
    fk_arm = fx +c1 * alpha * (gradfx @ direcx)
    return fk_arm


if __name__ == '__main__':

    n_val = 10**3

    # stablishing x_0 in  n_val
    x0 = np.random.default_rng(seed=42).uniform(-5, 5, n_val)

    #DEFINING THE VECTOR DE PRUEBA

    #vector_center = np.zeros(s_arr)
    #vector_boundaries = np.repeat(limit_values, s_arr, axis=1)
    #print(vector_boundaries)

    #VARIABLES NEEDED FOR THE PROCESS
    #number max of iterations
    kmax = 1000
    btmax = 100
    tolerance = 1e-12
    gamma = 1e-1
    c1 = 1e-4
    rho = 0.8

    l = 12

    # Defining the dominio
    #INIZIALIZATIONS FOR THE PROCESS:

    k=0 #number of iterations

    #checking if the x0 is out of bounds

    #defining the values of the function for x_PRUEBA
    f_xk = weighted_sphere_function(x0, n_val)

    #definig the exact gradient:
    g_xk= approx_gradient(x0, l, f_xk)

    gradfk_norm = np.linalg.norm(g_xk)
    deltaxk_norm = tolerance + 1

    xk = x0
    xk = Pi_x(x0)
    while k < kmax and gradfk_norm >= tolerance and deltaxk_norm >= tolerance:
        pk = -approx_gradient(xk, l, f_xk)
        xstepk = xk + gamma * pk
        xbark = Pi_x(xstepk)

        alpha = 1
        direc_xk = xbark - xk
        x_def = xk + alpha * direc_xk

        f_def = weighted_sphere_function(x_def, n_val)

        bt = 0
        f_arm = f_armijo_condition(f_xk, g_xk, c1, alpha, direc_xk)

        while bt < btmax and f_def > f_arm:
            alpha = rho * alpha
            x_def = xk + alpha * direc_xk
            f_def = weighted_sphere_function(x_def, n_val)

            bt=bt+1

        deltaxk_norm = np.linalg.norm(x_def - xk)
        xk = x_def
        f_xk = f_def

        g_xk = approx_gradient(xk, l, f_xk)
        gradfk_norm = np.linalg.norm(g_xk)
        k = k + 1

        #xseq[k:] = xk
        #btseq[:k] = bt

    #xseq = xseq[1:k, :]
    #btseq = btseq[:, 1:k]

    print(f"Last iteration: {xk}")
    print(f"Number of iterations: {k}, norm gradient: {gradfk_norm}, value function f(x)= {f_xk}")
    #print(f"Matrix of x:{xseq}")

