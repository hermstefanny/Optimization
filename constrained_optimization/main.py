
import numpy as np
import matplotlib as mt


def weighted_sphere_function(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    weighted_value = (x * natural_numbers) @ np.transpose(x)
    return weighted_value


def exact_gradient(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    eg = 2 * x * natural_numbers
    return eg


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

    #VARIABLES NEEDED FOR THE PROCESS
    #number max of iterations
    kmax = 1000
    btmax = 100
    tolerance = 1e-12
    gamma = 1e-1
    c1 = 1e-4
    rho = 0.8

    # Defining the dominio

    #INIZIALIZATIONS FOR THE PROCESS:

    k=0 #number of iterations

    #defining the values of the function for x_PRUEBA
    f_xk = weighted_sphere_function(x0, n_val)

    #definig the exact gradient:
    g_xk = exact_gradient(x0, n_val)

    gradfk_norm = np.linalg.norm(g_xk)
    deltaxk_norm = tolerance + 1

    xk = x0
    # checking if the x0 is out of bounds
    xk = Pi_x(x0)

    while k < kmax and gradfk_norm >= tolerance and deltaxk_norm >= tolerance:
        pk = -exact_gradient(xk, n_val)
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

        g_xk = exact_gradient(xk, n_val)
        gradfk_norm = np.linalg.norm(g_xk)
        k = k + 1

        #xseq[k:] = xk
        #btseq[:k] = bt

    #xseq = xseq[1:k, :]
    #btseq = btseq[:, 1:k]

    print(f"Last iteration: {xk}")
    print(f"Number of iterations: {k}, norm gradient: {gradfk_norm}, value function f(x)= {f_xk}")
    #print(f"Matrix of x:{xseq}")

