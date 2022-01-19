# This is a sample Python script.
import numpy as np
import matplotlib as mt


def weighted_sphere_function(x):
    weighted_value = 0
    for j, k in enumerate(x):
        weighted_value = weighted_value + ((j+1) * ((k) ** 2))
    return weighted_value

def exact_gradient(x):
    i = x.shape
    grad_vector = np.zeros(i)
    for j, k in enumerate(x):
        grad_vector[j] = (2 * (j+1) * k)
    return grad_vector

def Pi_x(x, lv):
    x_projected = np.zeros(np.shape(x)[0])
    for j, k in enumerate(x):
        if k < lv[0]:
            x_projected[j] = lv[0]
        elif k > lv[1]:
             x_projected[j] = lv[1]
        else:
            x_projected[j] = x[j]
    return x_projected

def f_armijo_condition(fx,  gradfx, c1, alpha, direcx):
    fk_arm = fx +c1 * alpha * (gradfx @ direcx)
    return fk_arm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #n = np.zeros(3)
    #d = np.array([3, 4, 5])
    #for i, j in enumerate(d):
        #n[i] = 10 ** j

    n_val = 10**2
    #stablishing x_0 in  n=10^3
    #x0 = np.linspace(1, 15, num=n[0].astype(np.int64))
    x0 = np.random.default_rng(seed=42).uniform(-100, 200, n_val)
    print(x0)

    #DEFINING THE VECTOR DE PRUEBA
    x_peque_prueba = x0 #np.array([154, -220, 40])

    s_arr = np.shape(x_peque_prueba)[0]
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

    # Defining the dominio
    limit_values = np.array([-5.12, 5.12])

    #INIZIALIZATIONS FOR THE PROCESS:
    #xseq = np.zeros((kmax, s_arr))
    #btseq = np.zeros((kmax, 1))
    k=0 #number of iterations

    #checking if the x0 is out of bounds
    proj_x = Pi_x(x_peque_prueba, limit_values)

    #defining the values of the function for x_PRUEBA
    f_xk = weighted_sphere_function(x_peque_prueba)

    #definig the exact gradient:
    g_xk= exact_gradient(x_peque_prueba)

    gradfk_norm = np.linalg.norm(g_xk)
    deltaxk_norm = tolerance + 1

    xk = x_peque_prueba
    xk = Pi_x(x_peque_prueba, limit_values)
    while k < kmax and gradfk_norm >= tolerance and deltaxk_norm >= tolerance:
        pk = -exact_gradient(xk)
        xstepk = xk + gamma * pk
        xbark = Pi_x(xstepk, limit_values)

        alpha = 1
        direc_xk = xbark - xk
        x_def = xk + alpha * direc_xk

        f_def = weighted_sphere_function(x_def)

        bt = 0
        f_arm = f_armijo_condition(f_xk, g_xk, c1, alpha, direc_xk)

        while bt < btmax and f_def > f_arm:
            alpha = rho * alpha
            x_def = xk + alpha * direc_xk
            f_def = weighted_sphere_function(x_def)

            bt=bt+1

        deltaxk_norm = np.linalg.norm(x_def - xk)
        xk = x_def
        f_xk = f_def

        g_xk = exact_gradient(xk)
        gradfk_norm = np.linalg.norm(g_xk)
        k = k + 1

        #xseq[k:] = xk
        #btseq[:k] = bt

    #xseq = xseq[1:k, :]
    #btseq = btseq[:, 1:k]

    print(f"Last iteration: {xk}")
    print(f"Number of iterations: {k}, norm gradient: {gradfk_norm}, value function f(x)= {f_xk}")
    #print(f"Matrix of x:{xseq}")

