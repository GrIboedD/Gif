import numpy as np
from console_progressbar import ProgressBar

def gradient(function, parameters, argument_increment=1e-06):
    """
    Алгоритм для вычисления градиента по формуле центральной разности для функций
    :param function: исходная функция
    :param parameters: параметры функции
    :param argument_increment: значение приращения аргумента (не обязательно)
    :return:
    """
    gradient = np.zeros_like(parameters)
    for i in range(len(parameters)):
        parameters_h_plus = parameters.copy()
        parameters_h_minus = parameters.copy()
        parameters_h_plus[i] += argument_increment
        parameters_h_minus[i] -= argument_increment
        gradient[i] = (function(*parameters_h_plus) - function(*parameters_h_minus)) / (2 * argument_increment)
    return gradient

def gradient_lf(loss_function, parameters, X, Y, argument_increment=1e-06):
    """
        Алгоритм для вычисления градиента по формуле центральной разности для функций потерь
        :param loss_function: исходная функция потерь
        :param parameters: параметры функции
        :param X: матрица входных данных
        :param Y: матрица выходных данных
        :param argument_increment: значение приращения аргумента (не обязательно)
        :return:
        """
    gradient = np.zeros_like(parameters)
    for i in range(len(parameters)):
        parameters_h_plus = parameters.copy()
        parameters_h_minus = parameters.copy()
        parameters_h_plus[i] += argument_increment
        parameters_h_minus[i] -= argument_increment
        gradient[i] = ((loss_function(*parameters_h_plus, X, Y) -
                        loss_function(*parameters_h_minus, X, Y))
                       / (2 * argument_increment))
    return gradient


def gradient_descent(function, initial_parameters, alpha, n_iter, epsilon=1e-06):
    """
    Алгоритм градиентного спуска для поиска локального минимума функции
    :param function: функция, минимум которой необходимо найти
    :param initial_parameters: начальные значения параметров
    :param alpha: скорость спуска
    :param n_iter: количество итераций
    :param epsilon: значение для остановки алгоритма, когда изменение по каждому параметру <= epsilon (не обязательно)
    :return: значение параметров в предполагаемом минимуме функции
    """
    parameters = initial_parameters.copy()
    for _ in range(n_iter):
        grad = gradient(function, parameters)
        difference = alpha * grad
        if np.all(np.abs(difference) <= epsilon):
            break
        parameters -= difference
    return parameters

def gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, X, Y, epsilon=1e-06):
    """
    Алгоритм градиентного спуска для поиска локального минимума функции потерь
    :param loss_function: функция потерь, минимум которой необходимо найти
    :param initial_parameters: начальные значения параметров
    :param alpha: скорость спуска
    :param n_iter: количество итераций
    :param X: матрица входных данных
    :param Y: матрица выходных данных
    :param epsilon: значение для остановки алгоритма, когда изменение по каждому параметру <= epsilon (не обязательно)
    :return: значение параметров в предполагаемом минимуме функции
    """
    parameters = initial_parameters.copy()
    print("Расчет оптимальных параметров методом градиентного спуска")
    pb = ProgressBar(total=n_iter-1, prefix='Progress', suffix='Complete', length=50)
    for i in range(n_iter):
        G = gradient_lf(loss_function, parameters, X, Y)
        difference = alpha * G
        if np.all(np.abs(difference) <= epsilon):
            pb.print_progress_bar(n_iter-1)
            print(f"Досрочный выход. n_iter = {i}")
            break
        parameters -= difference
        pb.print_progress_bar(i)
    return parameters


# def stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, X, Y, epsilon=1e-06, seed = 0):
#     """
#     Алгоритм градиентного спуска для поиска локального минимума функции потерь
#     :param loss_function: функция потерь, минимум которой необходимо найти
#     :param initial_parameters: начальные значения параметров
#     :param alpha: скорость спуска
#     :param n_iter: количество итераций
#     :param X: матрица входных данных
#     :param Y: матрица выходных данных
#     :param epsilon: значение для остановки алгоритма, когда изменение по каждому параметру <= epsilon (не обязательно)
#     :param seed: значения для инициализации генератора случайных чисел (0)
#     :return: значение параметров в предполагаемом минимуме функции
#     """
#     parameters = initial_parameters.copy()
#     best_loss = loss_function(*parameters, X, Y)
#     best_parameters = parameters.copy()
#     rng = np.random.default_rng(seed)
#     print("Расчет оптимальных параметров методом стохастического градиентного спуска")
#     pb = ProgressBar(total=n_iter-1, prefix='Progress', suffix='Complete', length=50)
#     for i in range(n_iter):
#         random_index = rng.integers(0, len(X))
#         G = gradient_lf(loss_function, parameters, X[random_index], Y[random_index])
#         difference = alpha * G
#         if np.all(np.abs(difference) <= epsilon):
#             pb.print_progress_bar(n_iter-1)
#             print(f"Досрочный выход. n_iter = {i}")
#             break
#         parameters -= difference
#         loss = loss_function(*parameters, X, Y)
#         if loss < best_loss:
#             best_loss = loss
#             best_parameters = parameters.copy()
#         pb.print_progress_bar(i)
#     return best_parameters

def stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, X, Y, epsilon=1e-06, max_n_iter_no_change = 10, seed = 0):
    """
    Алгоритм градиентного спуска для поиска локального минимума функции потерь
    :param loss_function: функция потерь, минимум которой необходимо найти
    :param initial_parameters: начальные значения параметров
    :param alpha: скорость спуска
    :param n_iter: количество итераций
    :param X: матрица входных данных
    :param Y: матрица выходных данных
    :param epsilon: значение для остановки алгоритма, когда изменение по каждому параметру <= epsilon (не обязательно)
    :param max_n_iter_no_change: максимальное количество итерации подряд, в течении которых функция потерь не уменьшается на значение больше или равное epsilon
    :param seed: значения для инициализации генератора случайных чисел (0)
    :return: значение параметров в предполагаемом минимуме функции
    """
    parameters = initial_parameters.copy()
    best_loss = loss_function(*parameters, X, Y)
    best_parameters = parameters.copy()
    rng = np.random.default_rng(seed)
    print("Расчет оптимальных параметров методом стохастического градиентного спуска")
    pb = ProgressBar(total=n_iter-1, prefix='Progress', suffix='Complete', length=50)
    n_iter_no_change = 0
    for i in range(n_iter):
        random_index = rng.integers(0, len(X))
        G = gradient_lf(loss_function, parameters, X[random_index], Y[random_index])
        difference = alpha * G
        parameters -= difference
        loss = loss_function(*parameters, X, Y)
        if loss + epsilon > best_loss:n_iter_no_change += 1
        else: n_iter_no_change = 0
        if n_iter_no_change >= max_n_iter_no_change:
            pb.print_progress_bar(n_iter - 1)
            print(f"Досрочный выход. n_iter = {i}")
            break
        if loss < best_loss:
            best_loss = loss
            best_parameters = parameters.copy()
        pb.print_progress_bar(i)
    return best_parameters

def minibatch_stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, X, Y, batch_size, epsilon=1e-06, max_n_iter_no_change = 10, seed = 0):
    """
    Алгоритм градиентного спуска для поиска локального минимума функции потерь
    :param loss_function: функция потерь, минимум которой необходимо найти
    :param initial_parameters: начальные значения параметров
    :param alpha: скорость спуска
    :param n_iter: количество итераций
    :param X: матрица входных данных
    :param Y: матрица выходных данных
    :param batch_size: размер одного пакета
    :param epsilon: значение для остановки алгоритма, когда изменение по каждому параметру <= epsilon (не обязательно)
    :param max_n_iter_no_change: максимальное количество итерации подряд, в течении которых функция потерь не уменьшается на значение больше или равное epsilon
    :param seed: значения для инициализации генератора случайных чисел (0)
    :return: значение параметров в предполагаемом минимуме функции
    """
    X = np.array(X)
    Y = np.array(Y)
    parameters = initial_parameters.copy()
    best_loss = loss_function(*parameters, X, Y)
    best_parameters = parameters.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], size=len(X), replace=False)
    shuffled_X = X[indices]
    shuffled_Y = Y[indices]
    start = 0
    print("Расчет оптимальных параметров методом мини-пакетного стохастического градиентного спуска")
    pb = ProgressBar(total=n_iter-1, prefix='Progress', suffix='Complete', length=50)
    n_iter_no_change = 0
    for i in range(n_iter):
        if start >= len(X):
            indices = rng.choice(X.shape[0], size=len(X), replace=False)
            shuffled_X = X[indices]
            shuffled_Y = Y[indices]
            start = 0
        end = min(len(X), start + batch_size)
        batch_X = shuffled_X[start:end]
        batch_Y = shuffled_Y[start:end]
        G = gradient_lf(loss_function, parameters, batch_X, batch_Y)
        difference = alpha * G
        parameters -= difference
        start += batch_size
        loss = loss_function(*parameters, X, Y)
        if loss + epsilon > best_loss:n_iter_no_change += 1
        else: n_iter_no_change = 0
        if n_iter_no_change >= max_n_iter_no_change:
            pb.print_progress_bar(n_iter - 1)
            print(f"Досрочный выход. n_iter = {i}")
            break
        if loss < best_loss:
            best_loss = loss
            best_parameters = parameters.copy()
        pb.print_progress_bar(i)
    return best_parameters
