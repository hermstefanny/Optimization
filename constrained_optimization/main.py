
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

# DEFINITIONS OF FUNCTIONS TO BE CALLED IN MAIN

# Function f(x):
def weighted_sphere_function(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    weighted_value = (x * natural_numbers) @ np.transpose(x)
    return weighted_value

# Function Exact Gradient:
def exact_gradient(x, dim):
    natural_numbers = np.arange(1, dim + 1)
    eg = 2 * x * natural_numbers
    return eg

# Function Projection of a point (x)
def Pi_x(x):
    x_projected = x
    x_projected[np.where(x < -5.12)] = -5.12
    x_projected[np.where(x > 5.12)] = 5.12
    return x_projected

# Function of the Armijo Condition
def f_armijo_condition(fx, grad_fx, sigma_k, alpha_k, direction):
    fk_arm = fx + sigma_k * alpha_k * (grad_fx @ direction)
    return fk_arm


# Main
if __name__ == '__main__':

    tic()

    # Number of values in vector x
    n_val = 10**4

    # INIZIALIZATIONS OF PARAMETERS FOR THE PROCESS:
    k_max = 100000
    bt_max = 100
    tolerance = 1e-12
    gamma = 1e-1
    sigma = 1e-4
    beta = 0.8
    alpha = 1

    # Number of data points to be stored in the table
    #data_points = int(k_max / 100)

    # Vectors to store values for tables
    # grad_norm_table = np.zeros(data_points)
    # f_k_table = np.zeros(data_points)
    # step_table = np.zeros(data_points)
    # num_points = np.zeros(data_points)
    # # Initial value for the vector of data points
    # i = 0

    # Vectors to store values for graphs
    grad_norm_vector_graph = np.zeros(k_max)
    f_k_vector_graph = np.zeros(k_max)
    step_vector_graph = np.zeros(k_max)
    bt_vector = np.zeros(k_max)

    # Generating random x_0 with seed:
    x0 = np.random.default_rng(seed=43).uniform(-10, 10, n_val)

    # Calling of function f(x) in x0:
    f_xk = weighted_sphere_function(x0, n_val)
    print(f'The value of the function in x0: {f_xk}')

    # Calling of function exact gradient in x0:
    g_xk = exact_gradient(x0, n_val)

    # Calculating the norm of the gradient in x0:
    grad_xk_norm = np.linalg.norm(g_xk)

    # Establishing the first step as:
    delta_xk_norm = tolerance + 1

    # Storing x0:
    x_k = x0

    # Checking if  x0 is out of bounds:
    x_k = Pi_x(x0)

    # initial number of iterations
    k = 0

    # OUTER ITERATIONS WITH WHILE LOOP
    while k < k_max and grad_xk_norm >= tolerance and delta_xk_norm >= tolerance:
        # Establishing the direction of the gradient:
        p_k = -exact_gradient(x_k, n_val)
        # Gradient step:
        x_hat_k = x_k + gamma * p_k
        # Projection step:
        x_bar_k = Pi_x(x_hat_k)
        # Direction of the projection:
        direc_x_k = x_bar_k - x_k
        # Calculating x_k+1 with not-reduced alpha
        x_next = x_k + alpha * direc_x_k
        # Value of function in x_k+1
        f_next = weighted_sphere_function(x_next, n_val)

        # INNER ITERATIONS
        # Calling of function of Armijo Condition for condition in while loop
        f_arm = f_armijo_condition(f_xk, g_xk, sigma, alpha, direc_x_k)

        # Inizialization of values for inner iterations
        bt = 0
        alpha_bt = alpha

        while bt < bt_max and f_next > f_arm:
            # Reducing the alpha:
            alpha_bt = beta * alpha_bt
            # Calculating x_k+1
            x_next = x_k + alpha_bt * direc_x_k
            # Calculating f(x_k+1)
            f_next = weighted_sphere_function(x_next, n_val)
            # Updating the condition
            f_arm = f_armijo_condition(f_xk, g_xk, sigma, alpha_bt, direc_x_k)
            # Increasing inner iterations
            bt = bt + 1

        # Saving the final number of the inner iteration
        bt_vector[k] = bt

        # Calculating the step with the norm of x_k+1 - x_k
        delta_xk_norm = np.linalg.norm(x_next - x_k)
        # Actualizing the values of x_k, f(x_k),  gradient in x_k and its norm
        x_k = x_next
        f_xk = f_next
        g_xk = exact_gradient(x_k, n_val)
        grad_xk_norm = np.linalg.norm(g_xk)

        # Saving values for the graphs:
        grad_norm_vector_graph[k] = grad_xk_norm
        f_k_vector_graph[k] = f_xk
        step_vector_graph[k] = delta_xk_norm

        # Increasing outer iterations:
        k = k + 1

        # Printing values to check if the method is working
        if k % 500 == 0:
            print(f' norm gradient: {grad_xk_norm}, '
                  f'value function f(x)= {f_xk}, '
                  f'value of step = {delta_xk_norm}')
            # num_points[i] = k
            # grad_norm_table[i] = grad_xk_norm
            # f_k_table[i] = f_xk
            # step_table[i] = delta_xk_norm
            #i = i+1

    # Cutting the vectors for the graphs to the last value of k iterations
    grad_norm_vector_graph = grad_norm_vector_graph[:k]
    f_k_vector_graph = f_k_vector_graph[:k]
    bt_vector = bt_vector[:k]

    # Calculating the max and min values of the last x_k
    max_value = np.max(x_k)
    min_value = np.min(x_k)

    # PRINTING LAST VALUES FOR ANALYSIS
    print(f"Number of total iterations: {k},  "
          f"Value of the last gradient norm: {grad_xk_norm}, "
          f"Last value of  function f(x)= {f_xk}, "
          f"Last value of  step = {delta_xk_norm}"
          f"The maximum value of the backtracking steps is {np.max(bt_vector)}")
    print(f"The individuals values of the last iteration are within: {min_value} and {max_value}")
    toc()

    # -------------------------- SECTION OF TABLES -----------------------------#

    #table_values = np.stack((num_points, f_k_table, grad_norm_table, step_table), axis=-1)
    #print(f"Table of Values {table_values}")
    # --------------------------- SECTION OF GRAPHS-----------------------------#
    # Calculating the values for x - axis:  from 0 to k iterations
    num_iterations = list(range(0, k))

    # Graph (1)
    plt.plot(num_iterations, grad_norm_vector_graph, '-', linewidth=0.5, c='g')
    plt.title("Norm of the gradient vs Iterations")
    plt.xlabel("Number of iterations")
    plt.ylabel("Norm of the gradient")
    plt.yscale('log')
    plt.grid()
    plt.savefig('grad_vs_step_gam_n10_4.png', bbox_inches='tight')
    plt.show()

    # Graph (2)
    plt.plot(num_iterations, f_k_vector_graph, '-', linewidth=1, c='b')
    plt.title("Value of the function vs Iterations")
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x)")
    plt.yscale('log')
    plt.grid()
    plt.savefig('function_vs_step_gam_n10_4.png', bbox_inches='tight')
    plt.show()

    # Graph (3)
    # plt.plot(num_iterations, step_vector_graph, '-', linewidth=1, c='r')
    # plt.title("Value of step vs Iterations")
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Step")
    # plt.yscale('log')
    # plt.grid()
    # plt.savefig('delta_vs_step_gam_n10_4.png', bbox_inches='tight')
    # plt.show()

