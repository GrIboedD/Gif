import matplotlib.pyplot as plt
import numpy as np
import two_step_optimization as tso
from functions import find_k
from optimization_2_param import generate_data, loss_function


def vis_loss_function(k0, en, x, y):
    return (find_k(k0, en, x) - y)**2

def visualize_k_loss(temp: float, k_real: float, range_k0: list, range_en: list, num_of_points: int):
    """
    Визуализация функции потерь
    :param temp: Температура
    :param range_k0: границы k0
    :param range_en: границы en
    :param num_of_points: количество точек
    """
    # Подготовка данных
    R = 1.987
    k0 = np.linspace(range_k0[0], range_k0[1], num_of_points)
    en = np.linspace(range_en[0], range_en[1], num_of_points)
    k0, en = np.meshgrid(k0, en)
    tso.generate_data(42, temp)
    k = (k0*np.exp(-en/(R * temp)) - k_real)**2
    # Построение графика
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(k0, en, k, cmap="viridis")
    ax.set_xlabel('k0')
    ax.set_ylabel('en')
    ax.set_zlabel('loss')
    plt.show()


def visualize_loss_function(range_k1: list, range_k2: list, num_of_points: int, ):
    # Подготовка данных
    k1_arr = np.linspace(range_k1[0], range_k1[1], num_of_points)
    k2_arr = np.linspace(range_k2[0], range_k2[1], num_of_points)
    k1_arr, k2_arr = np.meshgrid(k1_arr, k2_arr)
    x, y = generate_data(42)
    loss_arr = loss_function(k1_arr, k2_arr, x, y)

    # Построение 3d графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.plot_surface(k1_arr, k2_arr, loss_arr, cmap = 'viridis')

    ax.view_init(elev=30, azim=90)
    plt.show()