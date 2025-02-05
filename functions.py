import math

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