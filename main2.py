import math
import numpy as np
import matplotlib.pyplot as plt
from SGD import gradient_descent_lf, stochastic_gradient_descent_lf, minibatch_stochastic_gradient_descent_lf
# Универсальная газовая постоянная
R = 1.987


def find_k(k0, en, temp):
    """
    Расчет константы скорости реакции
    :param k0: пред экспоненциальный фактор
    :param en: энергия активации
    :param temp: температура
    :returns: константа скорости реакции для заданных значений
    """
    return k0 * math.exp(-en / (R * temp))

# Метод Рунге-Кутты четвертого порядка для вычисления текущей концентрации глюкозы

def find_g(k0_1, en1, k0_2, en2, g0, temp, time, h=1):
    """
    Расчет концентрации глюкозы в момент времени t
    :param k0_1: пред экспоненциальный фактор для глюкозы
    :param en1: энергия активации для глюкозы
    :param k0_2: пред экспоненциальный фактор для фруктозы
    :param en2: энергия активации для фруктозы
    :param g0: начальная концентрация глюкозы при t = 0
    :param temp: температура при которой происходит реакция
    :param time: момент времени t
    :param h: шаг метода Рунге-Кутты
    :returns: концентрацию глюкозы в момент времени t
    """
    # Функция расчета производной концентрации глюкозы по времени
    def f(g):
        k2 = find_k(k0_2, en2, temp)
        k1 = find_k(k0_1, en1, temp)
        return k2 * (g0 - g) - k1 * g
    # Нулевой момент времени
    t_i = 0
    # Значение концентрации глюкозы в момент времени t_1
    G_i = g0
    # Цикл вычисления концентрации глюкозы в момент времени t
    while t_i < time:
        t_i += h
        # промежуточный расчет
        K1 = f(G_i)
        K2 = f(G_i + h/2*K1)
        K3 = f(G_i + h/2*K2)
        K4 = f(G_i + h*K3)
        # значение концентрации для t_i + h
        G_i += h/6*(K1+2*K2+2*K3+K4)
    return G_i

def loss_function(k0_1, en1, k0_2, en2, x, y):
    """
    Функция потерь для модели
    :param k0_1: пред экспоненциальный фактор для глюкозы
    :param en1: энергия активации для глюкозы
    :param k0_2: пред экспоненциальный фактор для фруктозы
    :param en2: энергия активации для фруктозы
    :param x: матрица входных данных, каждая строчка содержит начальную концентрацию глюкозы, температуру и время
    :param y: вектор экспериментальных данных, концентраций глюкозы
    :return: Сумма квадратов разностей расчетной концентрации глюкозы и экспериментальной
    """
    # Инициализация потерь
    loss = 0
    # Расчет потерь
    if isinstance(y, (float, int)): return (y - find_g(k0_1, en1, k0_2, en2, *x)) ** 2
    for i in range(len(x)):
        loss += (y[i] - find_g(k0_1, en1, k0_2, en2, *x[i])) ** 2
    # Очень важно, чтобы величина функции потерь не зависела от количества входных данных. Для этого, например, можно вычислять среднее суммы
    # Иначе параметр alpha (скорость спуска) будет необходимо подбирать, например, отдельно, для каждого размера пакета
    return loss/len(x)

def generate_data(seed):
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
    !for g0 in [10, 15, 20]:
        time = 0
        while time <= 40:
            x.append([g0, 323.15, time])
            g = find_g(k0_1, en1, k0_2, en2, g0, time)
            # имитируем погрешность измерений в 5 процентов максимум
            g += g * np.random.uniform(-0.05, 0.05)
            y.append(g)
            # увеличиваем время на случайное число от 1 до 5
            time += np.random.randint(1, 6)
    return x, y


def main():
    # Проверка работы метода Рунге-Кутты
    g = find_g(1, 0, 1, 0, 20, 20, 40)
    print("Если допустить, что k1 и k2 = 1, то G -> G0/2 (производная в такой точке будет равна 0)")
    print(f"G0 = 20, G(40) = {g:.3}\n")
    x, y = generate_data(42)
    real_parameters = [5.91e5, 10733, 2.07, 2224]
    real_lost = loss_function(*real_parameters, x, y)
    print(f"\nЭталонные параметры: k0_1 = {real_parameters[0]:.3e}, en1 = {real_parameters[1]:.3e}, k0_2 = {real_parameters[2]:.3e}, en2 = {real_parameters[3]:.3e}")
    print(f"Функция потерь для эталонных параметров: {real_lost:.3}")

    initial_parameters = [1, 1, 1, 1]
    alpha = 0.004
    n_iter = 1000
    batch_size = 20
    n_no_change = 1000

    parameters_gd = gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y)
    lost_gd = loss_function(*parameters_gd, x, y)
    print(f"\nОптимизированные параметры: k0_1 = {parameters_gd[0]:.3e}, en1 = {parameters_gd[1]:.3e}, k0_2 = {parameters_gd[2]:.3e}, en2 = {parameters_gd[3]:.3e}")
    print(f"Функция потерь для оптимизированных параметров: {lost_gd:.3}\n")
    print(f"Дельта: {lost_gd - real_lost:.3}\n")

    parameters_sgd = stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y, max_n_iter_no_change=n_no_change)
    lost_sgd = loss_function(*parameters_sgd, x, y)
    print(f"\nОптимизированные параметры: k0_1 = {parameters_sgd[0]:.3e}, en1 = {parameters_sgd[1]:.3e}, k0_2 = {parameters_sgd[2]:.3e}, en2 = {parameters_sgd[3]:.3e}")
    print(f"Функция потерь для оптимизированных параметров: {lost_sgd:.3}")
    print(f"Дельта: {lost_sgd - real_lost:.3}\n")

    parameters_mbsgd = minibatch_stochastic_gradient_descent_lf(loss_function, initial_parameters, alpha, n_iter, x, y, batch_size, max_n_iter_no_change=n_no_change)
    lost_mbsgd = loss_function(*parameters_mbsgd, x, y)
    print(f"\nОптимизированные параметры: k0_1 = {parameters_mbsgd[0]:.3e}, en1 = {parameters_mbsgd[1]:.3e}, k0_2 = {parameters_mbsgd[2]:.3e}, en2 = {parameters_mbsgd[3]:.3e}")
    print(f"Функция потерь для оптимизированных параметров: {lost_mbsgd:.3}")
    print(f"Дельта: {lost_mbsgd - real_lost:.3}\n")

    # # Построение графиков
    # t_10 = []
    # g_10 = []
    # t_15 = []
    # g_15 = []
    # t_20 = []
    # g_20 = []
    # for i in range(len(x)):
    #     if x[i][0] == 10:
    #         t_10.append(x[i][2])
    #         g_10.append(y[i])
    #     elif x[i][0] == 15:
    #         t_15.append(x[i][2])
    #         g_15.append(y[i])
    #     else:
    #         t_20.append(x[i][2])
    #         g_20.append(y[i])
    # fig, ax = plt.subplots()
    # ax.scatter(t_10, g_10, c="b", s=30)
    # ax.scatter(t_15, g_15, c="r", s=30)
    # ax.scatter(t_20, g_20, c="g", s=30)
    # t = np.linspace(0, 45, 100)
    # g_10_gd = []
    # g_15_gd = []
    # g_20_gd = []
    # g_10_sgd = []
    # g_15_sgd = []
    # g_20_sgd = []
    # g_10_mbsgd = []
    # g_15_mbsgd = []
    # g_20_mbsgd = []
    # for i in range(len(t)):
    #     g_10_gd.append(find_g(*parameters_gd, 10, 323.15, t[i]))
    #     g_15_gd.append(find_g(*parameters_gd, 15, 323.15, t[i]))
    #     g_20_gd.append(find_g(*parameters_gd, 20, 323.15, t[i]))
    #
    #     g_10_sgd.append(find_g(*parameters_sgd, 10, 323.15, t[i]))
    #     g_15_sgd.append(find_g(*parameters_sgd, 15, 323.15, t[i]))
    #     g_20_sgd.append(find_g(*parameters_sgd, 20, 323.15, t[i]))
    #
    #     g_10_mbsgd.append(find_g(*parameters_mbsgd, 10, 323.15, t[i]))
    #     g_15_mbsgd.append(find_g(*parameters_mbsgd, 15, 323.15, t[i]))
    #     g_20_mbsgd.append(find_g(*parameters_mbsgd, 20, 323.15, t[i]))
    #
    # plt.plot(t, g_10_gd, color='r', linewidth=2)
    # plt.plot(t, g_15_gd, color='r', linewidth=2)
    # plt.plot(t, g_20_gd, color='r', linewidth=2)
    #
    # plt.plot(t, g_10_sgd, color='b', linewidth=2)
    # plt.plot(t, g_15_sgd, color='b', linewidth=2)
    # plt.plot(t, g_20_sgd, color='b', linewidth=2)
    #
    # plt.plot(t, g_10_mbsgd, color='g', linewidth=2)
    # plt.plot(t, g_15_mbsgd, color='g', linewidth=2)
    # plt.plot(t, g_20_mbsgd, color='g', linewidth=2)
    #
    # ax.set(xlim=[0, 45], ylim=[0, 25], xlabel='time', ylabel='G')
    # plt.show()