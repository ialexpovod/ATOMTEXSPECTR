# Парсинг файл-спектра
import os
import warnings
import datetime
import numpy
from . import plotting
from utility import sqrt_pitchs, pitch_centers_from_edges, handle_datetime, handle_uncertain, ALL_UFloats, machineEpsilon
import read


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
            uncerts = None,
            pitch_edges_keV = None,
            pitch_edges_raw = None,
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
            if uncerts is None and numpy.any(numpy.asarray(counts) < 0):
                raise SpectrERROR('Отрицательные значения, которые есть в отсчетах. '
                                  'Неопределенности, скорее всего, не распределены по закону Пуассона. '
                                  )
            # Теперь, когда с основами разобрались, стоит перейти к лямбда. Лямбда в Python — это просто еще один способ
            # определения функции. Вот базовый синтаксис лямбда-функции в Python:
            # lambda arguments: expression
            # >>> f = lambda x: x * x
            # >>> type(f)
            # <class 'function'>
            self._counts = handle_uncertain(counts, uncerts, lambda x: numpy.maximum(numpy.sqrt(x), 1))
        else:
            if len(cps) == 0:
                raise SpectrERROR("Спектр без скорости счета")
            self._cps = handle_uncertain(cps, uncerts, lambda x: numpy.nan) # numpy.nan -> nan (not None)

        if pitch_edges_raw is None and not (counts is None and cps is None):
            pitch_edges_raw = numpy.arange(len(self) + 1)
        self.pitch_edges_raw = pitch_edges_raw
        self.pitch_edges_keV = pitch_edges_keV

    @classmethod
    def import_file(cls, filename, debbuging = False):
        '''
        Метод создает объект спектра.

        :param filename:        строчный формат данных, представляющйи путь к файлу, подлежащему анализу
        :param debbuging:       Опциональный (необязательный), следует выводит отладочную информацию
        :return:                Объект-спектр
        '''
        _, extension = os.path.splitext(filename)
        if extension.lower() == ".spe":
            data = read.spe.reading(filename, debbuging = debbuging)
        elif extension.lower() == ".ats":
            data = read.ats.reading(filename, debbuging = debbuging)
        else:
            raise NotImplementedError(f"Разрешение файла {extension} не читабельно.")

        # создание объекта и применение калибровок
        spec = cls(**data)
        spec.attrs["infilename"] = filename
        # if cal is not None:
        #     spec.apply_calibration(cal)
        return spec