import numpy as np
from console_progressbar import ProgressBar
from SGD import minibatch_stochastic_gradient_descent_lf
from optimization_4_param import generate_data, loss_function
import warnings

def global_optimization(start, end, num_of_points):
    e1 = np.linspace(start, end, num_of_points)
    e2 = np.linspace(start, end, num_of_points)
    k0_1 = np.linspace(start, end, num_of_points)
    k0_2 = np.linspace(start, end, num_of_points)
    x, y = generate_data(42)
    loss_arr = []
    parameters_arr = []
    pb = ProgressBar(total=num_of_points**4-1,prefix="Progress", suffix='Complete', length=50)
    count = 0
    warnings.simplefilter('error', RuntimeWarning)
    alpha = 0.0001
    n_iter = 5
    n_iter_no_change = 5
    batch_size = 20
    for i in range(num_of_points):
        for j in range(num_of_points):
            for k in range(num_of_points):
                for l in range(num_of_points):
                    initial_parameters = [k0_1[i], e1[j], k0_2[k], e2[l]]
                    try:
                        parameters_mbsgd = minibatch_stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y, batch_size, max_n_iter_no_change=n_iter_no_change, suppress_stdout=True)
                        loss = loss_function(*parameters_mbsgd, x, y)
                        loss_arr.append(loss)
                        parameters_arr.append(parameters_mbsgd)
                    except RuntimeWarning:
                        pass
                    finally:
                        pb.print_progress_bar(count)
                        count += 1

    loss_arr = np.array(loss_arr)
    loss = np.min(loss_arr)
    i_min = np.argmin(loss_arr)
    parameters = parameters_arr[i_min]
    print("минималки")
    print(parameters)
    print(loss)
    n_iter_no_change = 100
    n_iter = 1000
    new_parameters = minibatch_stochastic_gradient_descent_lf(loss_function, parameters, alpha, n_iter, x, y, batch_size, max_n_iter_no_change=n_iter_no_change)
    loss = loss_function(*new_parameters, x, y)
    print("уточнил")
    print(new_parameters)
    print(loss)


