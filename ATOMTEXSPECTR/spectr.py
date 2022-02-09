# Парсинг файл-спектра
import os
import warnings
import datetime
import numpy
from . import plotting

class SpectrERROR(Exception):
    '''
        Исключение, полученное из Spectr.
    '''
    pass

class SpectrWarningg(UserWarning):
    '''
        Предупреждение изображения Spectr..
    '''

class UncalibratedError(SpectrERROR):
    '''
        Raise, некалиброванный спектр рассматривается как откалиброванный.
    '''
    pass


class Spectr:


    def __init__(
            self,
            counts = None,
            cps = None,
            uncs = None,
            bin_edges_keV = None,
            bin_edges_raw = None,
            livetime = None,
            realtime = None,
            start_time = None,
            stop_time = None,
            **kwargs,

    ):
        if not (counts is None) ^ (cps is None):
            raise SpectrERROR('Нужно указать counts или cps!')

        self._counts = None
        self._cps = None
        self._bin_edges_kev = None
        self._bin_edges_raw = None
        self.energy_cal = None
        self.livetime = None
        self.realtime = None
        self.attrs = {}

        if counts is not None:
            if (counts) == 0:
                raise SpectrERROR('Нет отсчетов файл-спектре.')
            if uncs is None and numpy.any(numpy.asarray(counts) < 0):
                raise SpectrERROR('Отрицательные значения, которые есть в отсчетах. '
                                  'Неопределенности, скорее всего, не распределены по закону Пуассона. '
                                  )
            self._counts = handle