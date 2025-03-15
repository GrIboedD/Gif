import optimization_2_param as o2p
import optimization_4_param as o4p
import numpy as np
from SGD import minibatch_stochastic_gradient_descent_lf, gradient_descent_lf
from functions import find_k
import math


def generate_data(seed, T):
    """
    Функция генерации данных
    :param seed: Значение инициализации рандома
    :return: матрица входных данных x [g0, temp, time] и вектор экспериментальных данных y [g]
    """
    # параметры, найденные в статье
    k0_1 = 5.91e5
    en1 = 10733
    k0_2 = 2.07
    en2 = 2224
    # инициализация гсч
    np.random.seed(seed)
    x = []
    y = []
    # цикл генерации данных
    for g0 in [10, 15, 20]:
        time = 0
        while time <= 40:
            x.append([g0, time])
            g = o4p.find_g(k0_1, en1, k0_2, en2, g0, T, time)
            # имитируем погрешность измерений в 5 процентов максимум
            g += g * np.random.uniform(-0.05, 0.05)
            y.append(g)
            # увеличиваем время на случайное число от 1 до 5
            time += np.random.randint(1, 6)
    return x, y

def loss_function(k0, en, x, y):
    loss = 0
    for i in range(len(x)):
        loss += (math.log(x) - y[i])**2
    loss /= len(x)
    return loss



def two_step_optimization():
    temperatures = [323.15, 333.15, 343.15]
    # k1_arr = []
    # k2_arr = []
    # for T in temperatures:
    #     x, y = generate_data(42, T)
    #     initial_parameters = [0, 0]
    #     alpha = 0.0001
    #     n_iter = 10
    #     batch_size = 15
    #     n_no_change = 100
    #     parameters_mbsgd = minibatch_stochastic_gradient_descent_lf(o2p.loss_function, initial_parameters, alpha, n_iter, x, y, batch_size, max_n_iter_no_change=n_no_change)
    #     k1_arr.append(parameters_mbsgd[0])
    #     k2_arr.append(parameters_mbsgd[1])

    initial_parameters = [15000, 0]
    alpha = 0.11
    n_iter = 10000
    parameters_gd = gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, temperatures, [0.0310, 0.0587, 0.0820])

    print("Мои ", loss_function(*parameters_gd, temperatures, [0.0310, 0.0587, 0.0820]))
    print("Их ", loss_function(5.91e5, 10733, temperatures, [0.0310, 0.0587, 0.0820]))



    print(parameters_gd)