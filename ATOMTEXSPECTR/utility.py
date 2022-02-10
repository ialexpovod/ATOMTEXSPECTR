'''
    Необходимые функции для основных модулей.
'''

import numpy
import datetime
from dateutil.parser import parse as dateutil_parse
# uncertainties -- модуль, позволяющий производить вычисления с включенными неопределенностями
# (погрешностями)
from uncertainties import UFloat, unumpy, ufloat
# x = ufloat(1, 0.1)
# >> x = 1 +/- 0.1
# print(2 * x)
# >> 2.00 +/- 0.2
# sin(2 * x)
# >> 0.9092974268256817+/-0.08322936730942848

import warnings
import matplotlib.pyplot
# В этой инициализации переменной мы задаем конкретный тип данных для машинного нуля (эпсилона)
# с плавающей запятой с помощью данной функции finfo()
def machineEpsilon(func = float):
    machine_epsilon = func(1)
    L = []
    while func(1)+func(machine_epsilon) != func(1):
        machine_epsilon_last = machine_epsilon
        L.append(machine_epsilon_last)
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last, L
# print(machineEpsilon()[1])
# matplotlib.pyplot.plot(numpy.linspace(0,10, len( machineEpsilon()[1])), machineEpsilon()[1])
# matplotlib.pyplot.show()
epsilon = machineEpsilon()[0]
# epsilon = numpy.finfo(float).eps
# >> 2.220446049250313e-16
# numpy.finfo(numpy.float64).eps
# >> 2.220446049250313e-16
# numpy.finfo(numpy.float32).eps
# >> 1.1920929e-07
# numpy.finfo(numpy.float16).eps
# >>0.000977

# Возможные векторы, представляющие информацию
vectors = (list, tuple, numpy.ndarray)

class UncertaintiesError(Exception):
    '''
        Raise, когда плохо указаны погрешности во входных данных
        (неточные данные погрешностей в input data).
    '''
    pass


def ALL_UFloats(x):
    '''
        Проверяет каждый элемент вектора x на принадлежность этого экземпляра
        к классу UFloat.
    :param x:   входной векто (проверяемый иттерационный объект).
    :return:    True все элементы x есть UFloats.
                False если элементы x есть не UFloats.
    '''
    # Добавим конструкцию исключения, как и при каком обстоятельстве может возникнуть
    # исключение. К экземпляру are_ufloats, с форматом данных - списки, происходит проверкаъ
    # каждого элемента в заданном списке списков на соответствие к данному объекту (Ufloat)
    #
    try:
        # isinstance() -- позволяет проверить принадлежность экземпляра к классу.
        # x = 'LoL', x - экземпляр, 'LoL' - объект.
        are_ufloats = [isinstance(item, UFloat) for item in x]  # instance x = [1,2,3] -> [False, False, False]
    except TypeError:
        return isinstance(x, UFloat)    # Если здесь False
    else:
        # Проверяем все ли элементы иттерационного объекта принадлежат
        # классу UFloat -- вернет True
        if all(are_ufloats): # все ли True
            return True
        # Если любой из элементов не соответствует требованию - какой-нибудь не
        # принадлежит классу UFloat -- вернет raise
        elif any(are_ufloats):
            raise UncertaintiesError("Входные компоненты объекта x должны"
                                     "все принадлежать классу UFloat "
                                     "или не принадлежать совсем.")
        else:
            return False
# x = [ufloat(1, 0.1), ufloat(1, 0.1)]
# print(ALL_UFloats(x))
# >> True
# print(isinstance(x, UFloat))

def handle_uncertain(x_vector, x_uncertainties, default_uncertaintience_function):
    '''
        Функция позволяет обрабатывать два метода определения неопределенностей (погрешностей).
        C floats или вручную.

    :param x_vector:                            list/tuple/array вектор, который может содержать UFloats

    :param x_uncertainties:                     list/tuple/array, который содержит собственные погрешности.

    :param default_uncertaintience_function:    функция, которая будет принимать в качестве входных даннных
                                                x_vector и возвращать набор значений x_uncertainties
                                                по умолчанию (еслиx_uncertainties не указан и x_vector не есть UFloat)

    :return:                                    numpy.array() of Ufloat

    '''

    ufloats = ALL_UFloats(x_vector)
    if ufloats and x_uncertainties is None:
        # x = True
        # x is not None
        # >> True
        # x is None
        # >> False
        return numpy.asarray(x_vector)  # a = [1,2,3] -> numpy.asarray(a) -> array([1,2,3])
    elif ufloats: # True
        raise UncertaintiesError(
            "Укажите значения погрешностей при помощи UFloats или"
            + "отдельным аргументом, но не сразу оба варианта."
        )
    elif x_uncertainties is not None:
        return unumpy.uarray(x_vector, x_uncertainties)
    else:
        return unumpy.uarray(x_vector, default_uncertaintience_function(x_vector))


def handle_datetime(input_time, allow_none = False):
    '''
        Парсинг компонентов даты, datetime, дата + время в строчном формате, или None.

        :param input_time:      входные данные, подлежащие преобразованию в дату.
        :param error_name:      имя, которое будет отображаться в случае возникновения ошибки.
        :param allow_none:      принимается ли значение 'None' в качестве входного и возвращаемого значения.

        :return:                datetime.datetime или None
    '''

    if isinstance(input_time, datetime.datetime):
        return input_time
    elif isinstance(input_time, datetime.date):
        warnings.warn(
            "datetime.date без времени; по умолчанию установлено значение 00:00 по дате"
        )
        return datetime.datetime(input_time.year, input_time.month, input_time.day)
    elif isinstance(input_time, str):
        return dateutil_parse(input_time)
    elif input_time is None and allow_none:
        return None
    else:
        raise TypeError(f"Неизвестный тип аргумента даты и времени: {input_time}")


def pitch_centers_from_edges(edges_kev):
    '''
        Вычисляет центр шага по назначенному диапазону.
    :param edges_kev:       иттерационный объект, представляющий как диапазон оси абсцисс.
    :return:                numpy.array() со значениями центров для каждого из шага в указанном диапазоне
    '''

    edges_kev = numpy.array(edges_kev)
    # x = numpy.array([1,2,3,4,5,6,7,8])
    # x[:-1]
    # >> array([1, 2, 3, 4, 5, 6, 7])
    # x[1:]
    # >> array([2, 3, 4, 5, 6, 7, 8])
    # (x[:-1] + x[1:]) / 2
    # >> array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
    return centers_kev


def sqrt_pitchs(pitch_edge_min, pitch_edge_max, npitchss):
    '''

    Корень квадратный из значений шага.

    :param pitch_edge_min:            Минимальное значение шага диапазона.
    :param pitch_edge_max:            Максимальное значение шага диапазона.
    :param npitchss:                   Число шагов.
    :return:                          numpy.array границ шага.
    '''

    """
    
    Args:
      bin_edge_min (float): Minimum bin edge (must be >= 0)
      bin_edge_max (float): Maximum bin edge (must be greater than bin_min)
      nbins (int): Number of bins
    Returns:
      np.array of bin edges (length = nbins + 1)
    """
    assert pitch_edge_min >= 0              # утверждение (инструкция, которая утверждает, что
                                            # минимальное значения шага в диапазоне должно быть больше
                                            # или равно нулю )
    assert pitch_edge_max > pitch_edge_min  # минимальное значение шага должно быть меньше
                                            # максимального в указанном диапазоне
    return numpy.linspace(numpy.sqrt(pitch_edge_min), numpy.sqrt(pitch_edge_max), npitchss + 1) ** 2


