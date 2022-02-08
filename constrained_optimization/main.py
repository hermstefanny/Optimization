
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import time

#CODE OBTAINED IN
#Code for counting the time
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
#

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


def f_armijo_condition(fx, grad_fx, sigma_k, alpha_k, direction):
    fk_arm = fx + sigma_k * alpha_k * (grad_fx @ direction)
    return fk_arm


if __name__ == '__main__':
    tic()
    n_val = 10**3

    # stablishing x_0 in  n_val
    x0 = np.random.default_rng(seed=43).uniform(-10, 10, n_val)

    #VARIABLES NEEDED FOR THE PROCESS
    #number max of iterations
    k_max = 1000
    bt_max = 100
    tolerance = 1e-12
    gamma = 1
    sigma = 1e-4
    beta = 0.8
    alpha = 1

    # Defining the domain
    #INIZIALIZATIONS FOR THE PROCESS:

    k = 0 #number of iterations

    #defining the values of the function for x_PRUEBA
    f_xk = weighted_sphere_function(x0, n_val)
    print(f'The value of the function in x0: {f_xk}')
    #definig the exact gradient:
    g_xk = exact_gradient(x0, n_val)

    grad_xk_norm = np.linalg.norm(g_xk)
    delta_xk_norm = tolerance + 1

    x_k = x0
    # checking if the x0 is out of bounds
    x_k = Pi_x(x0)

    grad_norm_vector = np.zeros(k_max)
    f_k_vector = np.zeros(k_max)
    bt_vector = np.zeros(k_max)

    while k < k_max and grad_xk_norm >= tolerance and delta_xk_norm >= tolerance:
        p_k = -exact_gradient(x_k, n_val)
        x_hat_k = x_k + gamma * p_k
        x_bar_k = Pi_x(x_hat_k)

        direc_x_k = x_bar_k - x_k
        x_next = x_k + alpha * direc_x_k
        f_next = weighted_sphere_function(x_next, n_val)

        f_arm = f_armijo_condition(f_xk, g_xk, sigma, alpha, direc_x_k)
        #print(f"In {k} iterations the armijo condition is {f_arm}")

        bt = 0
        alpha_bt = alpha
        while bt < bt_max and f_next > f_arm:
            alpha_bt = beta * alpha_bt
            x_next = x_k + alpha_bt * direc_x_k
            f_next = weighted_sphere_function(x_next, n_val)

            f_arm = f_armijo_condition(f_xk, g_xk, sigma, alpha_bt, direc_x_k)
            bt = bt + 1
        bt_vector[k] = bt

        delta_xk_norm = np.linalg.norm(x_next - x_k)
        x_k = x_next
        f_xk = f_next

        g_xk = exact_gradient(x_k, n_val)
        grad_xk_norm = np.linalg.norm(g_xk)

        grad_norm_vector[k] = grad_xk_norm
        f_k_vector[k] = f_xk

        k = k + 1

        if k % 50 == 0:
            print(f' norm gradient: {grad_xk_norm}, value function f(x)= {f_xk}, value of step = {delta_xk_norm}')

        #xseq[k:] = xk

    #xseq = xseq[1:k, :]
    #btseq = btseq[:, 1:k]

    grad_norm_vector = grad_norm_vector[:k]
    f_k_vector = f_k_vector[:k]
    bt_vector = bt_vector[:k]

    num_iterations = list(range(0, k))

    max_value = np.max(x_k)
    min_value = np.min(x_k)
    #print(f"Last iteration: {x_k}")
    print(f"Backtracking vector {bt_vector}")
    print(f"Number of iterations: {k},  "
          f"last norm gradient: {grad_xk_norm}, last value function f(x)= {f_xk}, last step = {delta_xk_norm}"
          f"The maximum value of the backtracking steps is {np.max(bt_vector)}")
    print(f"The individuals values of the last iteration are within: {min_value} and {max_value}")
    toc()

    plt.plot(num_iterations, grad_norm_vector, '-', linewidth=0.5, c= 'g')
    plt.title("Norm of the gradient vs Step")
    plt.xlabel("Number of iterations")
    plt.ylabel("Norm of the gradient")
    plt.yscale('log')
    plt.grid()
    plt.savefig('grad_vs_step_gam_1.png', bbox_inches='tight')
    plt.show()


    plt.plot(num_iterations, f_k_vector, '-',linewidth=1,  c='b')
    plt.title("Value of the function vs Step")
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x)")
    plt.yscale('log')
    plt.grid()
    plt.savefig('function_vs_step_gam_1.png', bbox_inches='tight')
    plt.show()

    #print(f"Matrix of x:{xseq}")

