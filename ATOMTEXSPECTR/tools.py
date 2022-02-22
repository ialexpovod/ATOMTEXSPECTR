"""
Required imported package functions.
"""
import numpy
import datetime
import warnings
from dateutil.parser import parse as dateutil_parse
from uncertainties import UFloat, unumpy
from warn import inaccuracy_error
# --------------- Initialization --------------- #
vectors = (list, tuple, numpy.ndarray)
# --------------- Inaccuracy funcs --------------- #
def ALL_UFloats(x) -> bool:
    try:
        are_ufloats = [isinstance(item, UFloat) for item in x]
    except TypeError:
        return isinstance(x, UFloat)
    else:
        if all(are_ufloats):
            return True
        elif any(are_ufloats):
            raise inaccuracy_error("Input should be all UFloats or no UFloats.")
        else:
            return False

def parsing_inaccuracy(
                        x,          # a list/tuple/numpy.ndarray that may contain UFloats
                        x_in,       # a list/tuple/numpy.ndarray that may contain manual uncertainty values
                        func        # a function that will take as input x_array and return a set of default
                                    # # values for x_uncs
                        ):
    """
    Parsing two methods of specifying error.
    """
    ufloats = ALL_UFloats(x)
    if ufloats and x_in is None:
        return numpy.asarray(x)
    elif ufloats: # True
        raise inaccuracy_error("Specify uncertainties with UFloats or "
                                + "by separate argument, but not both")
    elif x_in is not None:
        return unumpy.uarray(x, x_in)
    else:
        return unumpy.uarray(x, func(x_in))

# --------------- Inaccuracy funcs --------------- #
def machineEpsilon(func = float):
    machine_epsilon = func(1)
    L = []
    while func(1)+func(machine_epsilon) != func(1):
        machine_epsilon_last = machine_epsilon
        L.append(machine_epsilon_last)
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last, L[-1]
epsilon = machineEpsilon()[1]

# --------------- Parsing date/time --------------- #
def parsing_datetime(
                        input_time,            # the input argument to be converted to a datetime.
                        submit_None = False     # whether a None is allowed as an input and return value.
                    ):
    """
    Parse an argument as a date, datetime, date+time string, or None.
    """
    if isinstance(input_time, str):
        return dateutil_parse(input_time)
    elif isinstance(input_time, datetime.datetime):
        return input_time
    elif isinstance(input_time, datetime.date):
        warnings.warn(
            "datetime.date passed time; "
            "defaulting to 0:00 on date"
        )
        return datetime.datetime(input_time.year,
                                 input_time.month,
                                 input_time.day)
    elif input_time is None and submit_None:
        return None
    else:
        raise TypeError(f"Неизвестный тип аргумента даты и времени: {input_time}")


# todo закинуть в класс модуля spectrum.py
def bin_centers_from_edges(edges_kev):
    edges_kev = numpy.array(edges_kev)
    centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2
    return centers_kev

def sqrt_bins(bin_edge_min, bin_edge_max, nbinss):
    assert bin_edge_min >= 0
    assert bin_edge_max > bin_edge_min
    return numpy.linspace(numpy.sqrt(bin_edge_min),
                          numpy.sqrt(bin_edge_max), nbinss + 1) ** 2
# todo закинуть в класс модуля spectrum.py