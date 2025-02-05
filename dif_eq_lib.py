from scipy.integrate import odeint
import numpy as np
from functions import find_k

def solve_dif_eq(g0, t, k1, k2):
    """
    Решение ДУ с помощью odeint.
    :param g0: Начальная концентрация.
    :param t: Время.
    :param k1: Константа скорости реакции 1.
    :param k2: Константа скорости реакции 2.
    :return: Значение концентрации.
    """
    t = [0, t]
    def dG_dt(g, t):
        return k2 * (g0 - g) - k1 * g

    g = odeint(dG_dt, g0, t)
    g = np.array(g).flatten()
    return g[1]

def find_g(k0_1, en1, k0_2, en2, g0, temp, time):
    """
    Расчет концентрации глюкозы в момент времени t
    :param k0_1: пред экспоненциальный фактор для глюкозы
    :param en1: энергия активации для глюкозы
    :param k0_2: пред экспоненциальный фактор для фруктозы
    :param en2: энергия активации для фруктозы
    :param g0: начальная концентрация глюкозы при t = 0
    :param temp: температура при которой происходит реакция
    :param time: момент времени t
    :returns: концентрацию глюкозы в момент времени t
    """
    k2 = find_k(k0_2, en2, temp)
    k1 = find_k(k0_1, en1, temp)
    G = solve_dif_eq(g0, time, k1, k2)
    return G