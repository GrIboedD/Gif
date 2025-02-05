# Импорт функций для поиска оптимальных значений k1 и k2
from optimization_2_param import find_g, generate_data, loss_function
# Импорт методов градиентного спуска
from SGD import gradient_descent_lf, stochastic_gradient_descent_lf, minibatch_stochastic_gradient_descent_lf
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Проверка работы метода Рунге-Кутта
    g = find_g(1, 1, 20, 40)
    print("Если допустить, что k1 и k2 = 1, то G -> G0/2 (производная в такой точке будет равна 0)")
    print(f"G0 = 20, G(40) = {g:.3}\n")
    x, y = generate_data(42)
    real_parameters = [0.0310, 0.0653]
    real_lost = loss_function(*real_parameters, x, y)
    print(f"\nЭталонные параметры: k1 = {real_parameters[0]:.3e}, k2 = {real_parameters[1]:.3e}")
    print(f"Функция потерь для эталонных параметров: {real_lost:.3}\n")

    initial_parameters = [0, 0]
    alpha = 0.00001
    n_iter = 1000
    batch_size = 1
    n_no_change = 100

    parameters_gd = gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y)
    lost_gd = loss_function(*parameters_gd, x, y)
    print(f"\nОптимизированные параметры: k1 = {parameters_gd[0]:.3e}, k2 = {parameters_gd[1]:.3e}")
    print(f"Функция потерь для оптимизированных параметров: {lost_gd:.3}")
    print(f"Дельта: {lost_gd - real_lost:.3}\n")

    parameters_sgd = stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y, max_n_iter_no_change=n_no_change)
    lost_sgd = loss_function(*parameters_sgd, x, y)
    print(f"\nОптимизированные параметры: k1 = {parameters_sgd[0]:.3e}, k2 = {parameters_sgd[1]:.3e}")
    print(f"Функция потерь для оптимизированных параметров: {lost_sgd:.3}")
    print(f"Дельта: {lost_sgd - real_lost:.3}\n")

    parameters_mbsgd = minibatch_stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y, batch_size, max_n_iter_no_change=n_no_change)
    lost_mbsgd = loss_function(*parameters_mbsgd, x, y)
    print(f"\nОптимизированные параметры: k1 = {parameters_mbsgd[0]:.3e}, k2 = {parameters_mbsgd[1]:.3e}")
    print(f"Функция потерь для оптимизированных параметров: {lost_mbsgd:.3}")
    print(f"Дельта: {lost_mbsgd - real_lost:.3}\n")

    # Построение графиков
    t_10 = []
    g_10 = []
    t_15 = []
    g_15 = []
    t_20 = []
    g_20 = []
    for i in range(len(x)):
        if x[i][0] == 10:
            t_10.append(x[i][1])
            g_10.append(y[i])
        elif x[i][0] == 15:
            t_15.append(x[i][1])
            g_15.append(y[i])
        else:
            t_20.append(x[i][1])
            g_20.append(y[i])
    fig, ax = plt.subplots()
    ax.scatter(t_10, g_10, c="b", s=30)
    ax.scatter(t_15, g_15, c="r", s=30)
    ax.scatter(t_20, g_20, c="g", s=30)
    t = np.linspace(0, 45, 100)
    g_10_gd = []
    g_15_gd = []
    g_20_gd = []
    g_10_sgd = []
    g_15_sgd = []
    g_20_sgd = []
    g_10_mbsgd = []
    g_15_mbsgd = []
    g_20_mbsgd = []
    for i in range(len(t)):
        g_10_gd.append(find_g(*parameters_gd, 10, t[i]))
        g_15_gd.append(find_g(*parameters_gd, 15, t[i]))
        g_20_gd.append(find_g(*parameters_gd, 20, t[i]))

        g_10_sgd.append(find_g(*parameters_sgd, 10, t[i]))
        g_15_sgd.append(find_g(*parameters_sgd, 15, t[i]))
        g_20_sgd.append(find_g(*parameters_sgd, 20, t[i]))

        g_10_mbsgd.append(find_g(*parameters_mbsgd, 10, t[i]))
        g_15_mbsgd.append(find_g(*parameters_mbsgd, 15, t[i]))
        g_20_mbsgd.append(find_g(*parameters_mbsgd, 20, t[i]))

    plt.plot(t, g_10_gd, color='r', linewidth=2)
    plt.plot(t, g_15_gd, color='r', linewidth=2)
    plt.plot(t, g_20_gd, color='r', linewidth=2)

    plt.plot(t, g_10_sgd, color='b', linewidth=2)
    plt.plot(t, g_15_sgd, color='b', linewidth=2)
    plt.plot(t, g_20_sgd, color='b', linewidth=2)

    plt.plot(t, g_10_mbsgd, color='g', linewidth=2)
    plt.plot(t, g_15_mbsgd, color='g', linewidth=2)
    plt.plot(t, g_20_mbsgd, color='g', linewidth=2)

    ax.set(xlim=[0, 45], ylim=[0, 25], xlabel='time', ylabel='G')
    plt.show()
main()