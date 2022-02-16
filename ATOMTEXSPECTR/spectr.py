# Парсинг файл-спектра
import os
import warnings
import datetime
import numpy
from ATOMTEXSPECTR import plotting
from ATOMTEXSPECTR.utility import sqrt_pitchs, pitch_centers_from_edges, handle_datetime, handle_uncertain, ALL_UFloats, machineEpsilon
import ATOMTEXSPECTR.read
from uncertainties import UFloat, unumpy


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


# todo: 1. Расписать подробно в этом классе методы и атрибуты.
# todo: 2. Сделать примеры для plotting и spectr
class Spectr:
    '''
    Класс редставляет собой дифференциальное энергетическое распределение - СПЕКТР.

    Пример создания объекта класса:
        : spectr_1 = ATOMTEXSPECTR.Spectr()

    '''
    def __init__ (
            self,
            counts = None,
            cps = None,
            uncerts = None,
            #
            pitch_edges_keV = None,
            pitch_edges_raw = None,
            # Время проведения измерения, которое считает сама программа по набору спектральнных данных.
            measuretime = None,

            # Время проведения измерения по реальным данным времени (по Гринвичу). Фиксируется значение
            # из прибора, на котором производится набор спектральных данных, или компьютера с подключенным
            # к нему блоком детектирования или прибором - спектрометром.
            # Instance, 200 second
            actualtime = None,
            # Фиксированное значение даты и времени (instance, 02/07/2022 08:30:0 - MM/DD/YYYY HH:MM:SS)
            point_start = None,
            # Время прекращения измерения (actualtime + point_start = 02/07/2022 08:33:20)
            point_stop = None,
            **kwargs ):

        if not (counts is None) ^ (cps is None):        # True ^ True >>> False -> if not False: >>> True,
                                                        # if False: >>> условие не выполняется
            # False ^ True >>> True, True ^ False >>> True, False ^ False >>> False
            # То есть, если заданна сразу две переменной как "counts", так и
            # "cps", то вы получите ошибку получения спектра. Если не заданны
            # значения совсем -- ошибка получения спектра.
            raise SpectrERROR('Нужно указать counts или cps!')

        self._counts = None
        self._cps = None
        self._bin_edges_kev = None
        self._bin_edges_raw = None
        self.energy_cal = None
        self.measuretime = None
        self.attrs = {}

        if counts is not None:
            if len(counts) == 0:
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

        if measuretime is not None:
            # Если в перевееной нулевое присвоение:
            # То присвоить собственному значению переменной вещественное значение (с плавающей точкой)
            self.measuretime = float(measuretime)

        if actualtime is not None: # NULL
            self.actualtime = float(actualtime)
            if measuretime is not None:
                if self.measuretime > self.actualtime:
                    raise ValueError(f"Время измерения {measuretime} не может быть "
                                     f"больше фактического времени {actualtime}.")
        self.point_start = handle_datetime(point_start)
        self.point_stop = handle_datetime(point_stop)

        if (
            self.actualtime is not None and
            self.point_start is not None and
            self.point_stop is not None
        ):
            raise SpectrERROR("Должно быть указано не больше двух аргументов из трех для"
                              "actualtime, point_start, point_stop")

        elif (self,point_start is not None and
            self.point_stop is not None
              ):
            if self.point_start > self.point_stop:
                raise ValueError(f"Время начала {point_start} набора спектра "
                                 f"должно быть больше времени окончания {point_stop}!")
            self.actualtime = (self.point_stop - self.point_start).total_seconds()
        elif self.actualtime is not None and self.point_start is not None:
            self.point_stop = self.point_start + datetime.timedelta(seconds = self.actualtime)
        elif self.actualtime is not None and self.point_stop is not None:
            self.point_start = self.point_stop - datetime.timedelta(seconds = self.actualtime)

        # Любые другие
        for k in kwargs:
            self.attrs[key] = kwargs[key]
        # todo fix __array_ufunc__
        # These two lines make sure operators between a Spectrum
        # and a numpy arrays are forbidden and cause a TypeError
        self.__array_ufunc__ = None
        self.__array_priority__ = 1






    @property
    def counts(self):
        if self._counts is not None:
            return self._counts
        else:
            try:
                return self.cps * self.meatime
            except TypeError:
                raise SpectrERROR\
                (
                    "Неизвестное время измерения; невозможно получить скорость счета из количества отсчетов"
                )
    @property
    def counts_vals(self):
        # todo задать описание метода
        return unumpy.nominal_values(self.counts)

    @property
    def counts_uncs(self):
        # todo задать описание метода
        return unumpy.std_devs(self.counts)

    @property
    def cps(self):
        # todo задать описание метода
        if self._cps is not None:
            return self._cps
        else:
            try:
                return self.counts / self.meatime
            except TypeError:
                raise SpectrERROR(
                    "Unknown meatime; cannot calculate CPS from counts"
                )

    @property
    def cps_vals(self):
        # todo задать описание метода
        return unumpy.nominal_values(self.cps)

    @property
    def cps_uncs(self):

        return unumpy.std_devs(self.cps)

    @property
    def cpskev(self):

        return self.cps / self.bin_widths_kev

    @property
    def cpskev_vals(self):

        return unumpy.nominal_values(self.cpskev)

    @property
    def cpskev_uncs(self):

        return unumpy.std_devs(self.cpskev)

    @property
    def channels(self):

        warnings.warn(
            "channels is deprecated terminology and will be removed "
            "in a future release. Use bin_indices instead.",
            DeprecationWarning,
        )
        return numpy.arange(len(self), dtype=int)

    @property
    def bin_centers_raw(self):

        return pitch_centers_from_edges(self.bin_edges_raw)

    @property
    def bin_widths_raw(self):
        '''
            Ширина каждого канала, в keV.
        '''
        return numpy.diff(self.bin_edges_raw)

    @property
    def bin_centers_kev(self):

        if not self.is_calibrated:
            raise UncalibratedError("Spectrum is not calibrated")
        else:
            return pitch_centers_from_edges(self.bin_edges_kev)

    @property
    def energies_kev(self):

        warnings.warn(
            "energies_kev is deprecated and will be removed in a "
            "future release. Use bin_centers_kev instead.",
            DeprecationWarning,
        )

        if not self.is_calibrated:
            raise UncalibratedError("Spectrum is not calibrated")
        else:
            return pitch_centers_from_edges(self.bin_edges_kev)

    @property
    def bin_widths_kev(self):

        if not self.is_calibrated:
            raise UncalibratedError("Spectrum is not calibrated")
        else:
            return numpy.diff(self.bin_edges_kev)

    @property
    def bin_widths(self):

        warnings.warn(
            "bin_widths is deprecated and will be removed in a "
            "future release. Use bin_widths_kev (or bin_widths_raw) "
            "instead.",
            DeprecationWarning,
        )

        if not self.is_calibrated:
            raise UncalibratedError("Spectrum is not calibrated")
        else:
            return numpy.diff(self.bin_edges_kev)

    @property
    def is_calibrated(self):

        return self.bin_edges_kev is not None

    @property
    def bin_edges_kev(self):

        return self._bin_edges_kev

    @bin_edges_kev.setter
    def bin_edges_kev(self, bin_edges_kev):

        if bin_edges_kev is None:
            self._bin_edges_kev = None
        elif len(bin_edges_kev) != len(self) + 1:
            raise SpectrERROR("Bad length of bin edges vector")
        elif numpy.any(numpy.diff(bin_edges_kev) <= 0):
            raise ValueError("Bin edge energies must be strictly increasing")
        else:
            self._bin_edges_kev = numpy.array(bin_edges_kev, dtype=float)

    @property
    def bin_edges_raw(self):

        return self._bin_edges_raw

    @bin_edges_raw.setter
    def bin_edges_raw(self, bin_edges_raw):

        if bin_edges_raw is None:
            self._bin_edges_raw = None
        elif len(bin_edges_raw) != len(self) + 1:
            raise SpectrERROR("Bad length of bin edges vector")
        elif numpy.any(numpy.diff(bin_edges_raw) <= 0):
            raise ValueError("Raw bin edges must be strictly increasing")
        else:
            self._bin_edges_raw = numpy.array(bin_edges_raw, dtype=float)


    # Методы класса. Классовые методы, которые можно вызвать не создавая экземпляры.
    # spectr_1 = Spectr.import_file(filename)
    # class method
    # Spectr.channels()
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # TypeError: channels() missing 1 required positional argument: 'self'

    @classmethod
    def import_file(cls, filename, debugging = False):
        '''
        Метод создает объект спектра.

        :param filename:        строчный формат данных, представляющйи путь к файлу, подлежащему анализу
        :param debbuging:       Опциональный (необязательный), следует выводит отладочную информацию
        :return:                Объект-спектр
        '''
        _, extension = os.path.splitext(filename)
        if extension.lower() == ".spe":
            data = ATOMTEXSPECTR.read.spe.reading(filename, debugging = debugging)
        elif extension.lower() == ".ats":
            data = ATOMTEXSPECTR.read.ats.reading(filename, debugging = debugging)
        else:
            raise NotImplementedError(f"Разрешение файла {extension} не читабельно.")

        # создание объекта и применение калибровок
        spec = cls(**data)
        spec.attrs["infilename"] = filename
        # if cal is not None:
        #     spec.apply_calibration(cal)
        return spec

    @classmethod
    def import_from_list(
            cls, data_list,
            channels = None,
            calibration = False,
            xmin = None,
            xmax = None,
            ):
        '''
        Этот метод создает объект СПЕКТР из массива даныыых для формата данных список - list().

        :param data_list:
        :param channels:
        :param calibration:
        :param xmin:
        :param xmax:
        :return:
        '''





    def plot(self, *args, **kwargs):
        '''
        Графическое построение спектра с помощью Matplotlib командой plot.
        :param          args:   matplotlib как plot в виде строкого типа данных.
                        xmode:  опредлить что откладывается на оси абсцисс ('energy' или 'канал').
                                По дефолту -- энергия если есть.
                        ymode:  определить что откладывается на оси ординат ('counts', 'cps').
                        xlim:   задать диапазон для оси абсцисс.
                        ylim:   задать диапазон для оси ординат.
                        ax:     matplotlib координатные оси, если не предусмотренно, то создается.
                        yscale: вид оси ординат: 'linear', 'log', 'logit', 'symlog'.
                        title:  собственное задание названия для спектра.
                        xlabel: собственное задание названия для оси абсцисс.
                        ylabel: собственное задание названия для оси ординат.
                        emode:  может быть задана "band" для добавления 'error band' или 'bars'
                                для добавления 'error bars', по дефолту "None". Он совпадает с цветом
                                отображаемого спектра matplotlib и не может быть настроен.
                                Для лучшего управления построением спектра необходимо использовать
                                PlotSpectrum и его функции 'error band' и 'error bars'.

        :param kwargs:
        :return:
        '''
        # kwargs = {'A': 1, 'B': 2}
        # kwargs.get('B', 2) >>> 2; kwargs.get('A', 1) >>> 1; kwargs >>> {'A': 1, 'B': 2}
        # kwargs.get('B', 1) >> 2; kwargs.get('A', 'None') >>> 1; kwargs >>> {'A': 1, 'B': 2}
        # kwargs.pop('B', 2) >>> 2; kwargs >>> {'A': 1}
        # kwargs.pop('A', 1) >>> 1; kwargs >>> {}
        # kwargs = {'A': 1, 'B': 2}
        # kwargs.pop('A', "None"); kwargs >>> {'B': 2}
        # kwargs = {'A': 1, 'B': 2}; kwargs.pop('C', 3) >>> {'A': 1, 'B': 2}
        # kwargs = {'A': 1, 'B': 2}; kwargs.get('C', 3) >>> {'A': 1, 'B': 2}
        # .pop() - удаляет и сначала применяет существющее потом заменяет
        # .get() - не удаляет, и каждое присвоение соответствует ьзначению из словаря

        emode = kwargs.pop("emode", "none") # пременная emode присваивает значение, содержащее
                                            # по ключу 'emode' в словаре (не "none"),
                                            # после чего удаляется из словаря
        # Присваеивается значение из словаря без его стирания из него
        alpha = kwargs.get("alpha", 1)      # присваивается значение без удаления из словаря
                                            # если не существует в словаре - присвоет, но не добавит в словарь
        plotaxes = plotting.PlotSpectrum(self, *args, **kwargs)
        ax = plotaxes.plot()
        color = ax.get_lines()[-1].get_color()
        if emode == 'band':
            plotaxes.errorband(color = color, alpha = alpha * 0.5, label = "_nolegend_" )
        elif emode == "bars" or emode == "bar":
            plotaxes.errorbar(color = color, label = "_nolegend_")
        elif emode != "none":
            raise SpectrERROR(f"Неизвестный формат задания погрешностей {emode}, "
                              f"используйте 'bars' или 'band' ")
        return ax