"""
----------------
Spectr - fundamental class for file-spectrum parsing [0].
----------------
[0]     This project originates from the matplotlib and NumPy libraries.
        The attributes that are assigned here are similar to the attributes
        from these libraries, this will make the code less complex and
        understandable in terms of describing methods.
"""
import os
import warnings
import datetime
import numpy
from ATOMTEXSPECTR import plotting
from ATOMTEXSPECTR.utility import sqrt_bins, bin_centers_from_edges, handle_datetime, handle_uncertain, ALL_UFloats, machineEpsilon
import ATOMTEXSPECTR
from uncertainties import UFloat, unumpy
"HELLO"
class SpectrERROR(Exception):
    """
    Исключение, полученное из Spectr.

    СОздание класса в Pythin ачинается с инструкции class. Вот так будет выглядеть минимальный класс.
    Который ничего не делаает - pass. При его объявлении возникает сообщение об исключении.
    """
    pass

class SpectrWarningg(UserWarning):
    '''
        Предупреждение отображения Spectr..
    '''

class UncalibratedError(SpectrERROR):
    '''
        Raise, некалиброванный спектр рассматривается как откалиброванный.
    '''
    pass



class Spectr:
    '''
    Класс редставляет собой дифференциальное энергетическое распределение - СПЕКТР.

    Пример создания объекта класса:
        : spectr_1 = ATOMTEXSPECTR.Spectr()

    '''

    # Классы содержат методы и атрибуты. Атрибуты бывают  статическими и динамическими. Дальнейшее объявление атрибутов -
    # динамические. Динамические объявляются в методе __init__.
    # Пример статического атрибута может быть объявлен прямо в теле класаа:
    # Cs_137 = '662 keV'
    # >>> Spectr.Cs_137
    # '662 keV'
    # >>> Spectr.Cs_137 = '661 keV'
    # "661 keV"
    # Разница лишь в том, что для атрибута динамического
    # необходимо создавать экземпляр класса:
    # self.name_exampler = name_exampler
    # Для доступа к этим арибутам необходимо задать экземпляр класса:
    # >>> sp = Spectr('Ok')
    # >>> sp.name_exampler
    # 'Ok'

    def __init__ (
            self,
            counts = None,
            cps = None,
            uncerts = None,
            #
            bin_edges_keV = None,
            bin_edges_raw = None,
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

        if bin_edges_raw is None and not (counts is None and cps is None):
            bin_edges_raw = numpy.arange(len(self) + 1)
        self.bin_edges_raw = bin_edges_raw
        self.bin_edges_keV = bin_edges_keV

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
        self.point_start = handle_datetime(point_start, allow_none = True)
        self.point_stop = handle_datetime(point_stop , allow_none = True)

        if (
            self.actualtime is not None and
            self.point_start is not None and
            self.point_stop is not None
        ):
            raise SpectrERROR("Должно быть указано не больше двух аргументов из трех для"
                              "actualtime, point_start, point_stop")

        elif (self.point_start is not None and
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

        # Добавить все другие элементы в словарь attrs().
        # 'sigma': array([ 0.      ,  0.117188,  0.238281, ..., 17.175781, 17.1875  ,
        #        17.199219]), 'filename': 'D:\\ATOMTEXSPECTR\\tests\\spectrum\\228th.spe'}
        # Если не будет, то { 'filename': 'D:\\ATOMTEXSPECTR\\tests\\spectrum\\228th.spe'}
        for key in kwargs:
            self.attrs[key] = kwargs[key]
        # todo
        # These two lines make sure operators between a Spectrum
        # and a numpy arrays are forbidden and cause a TypeError
        self.__array_ufunc__ = None
        self.__array_priority__ = 1

    def __str__(self):

        # Специальные методы __str__ и __repr__ отвечают
        # за строковое представления объекта. При этом используются они в разных местах.

        # class IPAddress:
        #    def __init__(self, ip):
        #       self.ip = ip
        #
        #    def __str__(self):
        #       return f"IPAddress: {self.ip}"

        # >>> ip1 = IPAddress('10.1.1.1')
        # >>> ip2 = IPAddress('10.2.2.2')
        #
        # >>> str(ip1)
        # 'IPAddress: 10.1.1.1'
        #
        # >>> str(ip2)
        # 'IPAddress: 10.2.2.2'
        
        lines = ["ATOMTEXSPECTR.Spectr"]
        ltups = []
        for index in ['point_start', 'point_stop', 'actualtime',  'measuretime', 'is_calibrated']:
            # getattr() Возвращает значение атрибута или значение по умолчанию, если первое не было указано
            ltups.append((index, getattr(self, index)))
        ltups.append(("Каналов", len(self.bin_indices)))
        if self._counts is None:
            ltups.append(("gross_counts", None))
        else:
            ltups.append(("Общее количество отсчетов", self.counts.sum()))
            try:
                ltups.append(("Суммареая скорость счета", self.cps.sum()))
            except SpectrERROR:
                ltups.append(("gross_cps", None))
            if "filename" in self.attrs:
                ltups.append(("filename", self.attrs["filename"]))
            else:
                ltups.append(("filename", None))
            for lt in ltups:
                lines.append("    {:40} {}".format(f"{lt[0]}:", lt[1]))
            return "\n".join(lines)
    # В нашем случае методы __repr__ и __str__ равны
    __repr__ = __str__

    @property
    def counts(self):
        if self._counts is not None:
            return self._counts
        else:
            try:
                return self.cps * self.measuretime
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
                return self.counts / self.measuretime
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
    def bin_indices(self):
        """Bin indices.
        Returns:
          np.array of int's from 0 to (len(self.counts) - 1)
        """

        return numpy.arange(len(self), dtype = int)

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

        return bin_centers_from_edges(self.bin_edges_raw)

    @property
    def bin_widths_raw(self):
        '''
            Ширина каждого канала, в keV.
        '''
        return numpy.diff(self.bin_edges_raw)

    @property
    def bin_centers_kev(self):

        if not self.is_calibrated:
            raise UncalibratedError("Spectr не клиброван")
        else:
            return bin_centers_from_edges(self.bin_edges_kev)

    @property
    def energies_kev(self):

        warnings.warn(
            "energies_kev is deprecated and will be removed in a "
            "future release. Use bin_centers_kev instead.",
            DeprecationWarning,
        )

        if not self.is_calibrated:
            raise UncalibratedError("Spectr не калиброван")
        else:
            return bin_centers_from_edges(self.bin_edges_kev)

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
        '''
        Здесь осуществляется проверка.
        :param bin_edges_kev:
        :return:
        '''
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


    # Методы класса. Метод - это функция, находящаяся внутри класса и выполняющая
    # определенную работу. Бываю методы: статическими, классовыми и уровнем класса.
    # CСтатический метод создается с декоратором @staticmethod
    # Классовый метод - @classmethod, первый, по умолчанию, аргумент - cls.
    # Обычный метод, или уровень класса создается без специального декоратора,
    # первый его аргумент - self.
    # Статический и классовый метод можно вызвать, не создавая экземпляр класса.

    # Классовые методы, которые можно вызвать не создавая экземпляры.
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
            data = ATOMTEXSPECTR.read.spe.reading(filename, deb = debugging)
        elif extension.lower() == ".ats":
            data = ATOMTEXSPECTR.read.ats.reading(filename, deb = debugging)
        else:
            raise NotImplementedError(f"Разрешение файла {extension} не читабельно.")

        # создание объекта и применение калибровок
        spec = cls(**data)
        spec.attrs["filename"] = filename
        # if cal is not None:
        #     spec.apply_calibration(cal)
        return spec






    @classmethod
    def import_from_list(
            cls, data_list,
            bins = None,
            calibration = False,
            xmin = None,
            xmax = None,
           **kwargs):
        '''
        Этот метод создает объект СПЕКТР из массива даныыых для формата данных список - list().

        :param data_list:
        :param channels:
        :param calibration:
        :param xmin:
        :param xmax:
        :return:
        '''

        assert len(data_list) > 0

        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = numpy.ceil(max(data_list))
        if bins is None:
            bins = numpy.arange(xmin, xmax + 1, dtype=int)

        assert xmin < xmax
        if isinstance(bins, int):
            assert bins > 0
        else:
            assert len(bins) > 1

        bin_counts, bin_edges = numpy.histogram(data_list, bins = bins, range = (xmin, xmax) )

        kwargs["counts"] = bin_counts
        kwargs["bin_edges_kev" if calibration else "bin_edges_raw"] = bin_edges

        return cls(**kwargs)

    def copy(self):
        from copy import deepcopy
        '''
            Делает deep copy этого объекта спектр.
        '''
        return deepcopy(self)

    def __len__(self):
        '''
        Число каналов (bins) объекта Spectr.
        :return: формат int для числа каналов
        '''
        try:
            return len(self.counts)
        except SpectrERROR:
            return len(self.cps)

    def __add__(self, other):
        '''
        Сложение спектров. Сумма времени измерений (если оно задано) и
        результирующего спектра (по распределению Пуассона).
        Оба спектра могу быть некалиброваными, или оба могут быть калиброванными
        с одинаковой калибровкой по энергии.

        :param other:       другой объект (экземпляр) Spectr для добавления отсчетов (count)
        :return:            суммированный объект Spectr.
        '''

        self._add_sub_error_checking(other)
        if (self._counts is None) ^ (other._counts is None):
            raise SpectrERROR(
                "Сложение спектров на отсчетах (counts), и спектров на скорости счета (cps) "
                + "два варианта, использовать Spectr(counts = specA.counts + specB.counts) "
                + "или Spectr(cps = specA.cps + specB.cps)."
            )

        if self._counts is not None and other._counts is not None:
            kwargs = {"counts": self.counts + other.counts}
            if self.measuretime and other.measuretime:
                kwargs["measuretime"] = self.measuretime + other.measuretime
            else:
                warnings.warn(
                    "Сложение отсчетов с ошибкой по времени измерения, "
                    + "время измерений есть None.",
                    SpectrWarningg,
                )
        else:
            kwargs = {"cps": self.cps + other.cps}

        if self.is_calibrated and other.is_calibrated:
            spect_obj = Spectr(bin_edges_kev = self.bin_edges_kev, **kwargs)
        else:
            spect_obj = Spectr(bin_edges_raw = self.bin_edges_raw, **kwargs)
        return spect_obj

    def __sub__(self, other):
        """
        Вычитание спектров. Результирующий спектр не имеет значимого время измерения или
        рассчитывает вектор и НЕ распределяется по Пуассону.
        Оба спектра могут быть некалиброванными или оба могут быть откалиброваны
        по той же энергии.

        :param other: другой объект Spectr.
        :return:      вычтенный спектр.
        """

        self._add_sub_error_checking(other)
        try:
            kwargs = {"cps": self.cps - other.cps}
            if (self._cps is None) or (other._cps is None):
                warnings.warn(
                    "Вычитание спектров на отсчетах (counts) , "
                    + "спектры были преобразовны в скорость счета (cps)",
                    SpectrWarningg,
                )
        except SpectrERROR:
            try:
                kwargs = {"counts": self.counts_vals - other.counts_vals}
                kwargs["uncs"] = [numpy.nan] * len(self)
                warnings.warn(
                    "Вычитание на отсчетах (counts), "
                    + "время измерения пропущенно.",
                    SpectrWarningg,
                )
            except SpectrERROR:
                raise SpectrERROR(
                    "Вычитание отсчетов и скорости счета спектров без "
                    + "времени измерения невозможно."
                )

        if self.is_calibrated and other.is_calibrated:
            spect_obj = Spectr(bin_edges_kev=self.bin_edges_kev, **kwargs)
        else:
            spect_obj = Spectr(bin_edges_raw=self.bin_edges_raw, **kwargs)
        return spect_obj


    def _add_sub_error_checking(self, other):
        '''
        Метод для обработки ошибок при сложении или вычитании спектров.
        :param other:   другой спектр.
        :return:
        '''

        if not isinstance(other, Spectr):
            raise TypeError(
                "Операция по сложению/вычитанию Spectr должно включать объект Spectr"
            )
        if len(self) != len(other):
            raise SpectrERROR("Нельзя складывать/вычитать спектры с различным количеством каналов ")
        if self.is_calibrated ^ other.is_calibrated:
            raise SpectrERROR(
                "Невозможно складывать/вычитать некалиброванные спектры с/от "
                + "калиброванным спектром. Если оба имеют одинаковую калибровку, "
                + 'используйте метод "calibrate_like"'
            )
        if self.is_calibrated and other.is_calibrated:
            if not numpy.all(self.bin_edges_kev == other.bin_edges_kev):
                raise NotImplementedError(
                    "Сложение/вычитание для произвольных калиброванных спектров"
                    + "не реализовано."
                )
        if not self.is_calibrated and not other.is_calibrated:
            if not numpy.all(self.bin_edges_raw == other.bin_edges_raw):
                raise NotImplementedError(
                    "Сложение/вычитание для произвольных некалиброванных спектров"
                    + "не реализовано."
                )

    def __mul__(self, other):
        """
         Возвращает новый объект Spectr с увеличенным количеством отсчетов (counts)
         или скорости счета (cps).
        :param other:   коэффициент умножения на объект Spectr.
        :return:        новый объект Spectr.
        """
        return self._mul_div(other, div = False)

    # This line adds the right multiplication
    __rmul__ = __mul__

    def __div__(self, other):
        """Return a new Spectrum object with counts (or CPS) scaled down.
        Args:
          factor: factor to divide by. May be a ufloat.
        Raises:
          TypeError: if factor is not a scalar value
          SpectrumError: if factor is 0 or infinite
        Returns:
          a new Spectrum object
        """

        return self._mul_div(other, div=True)

    # This line adds true division
    __truediv__ = __div__

    def _mul_div(self, scaling_factor, div=False):
        """Multiply or divide a spectrum by a scalar. Handle errors.
        Raises:
          TypeError: if factor is not a scalar value
          ValueError: if factor is 0 or infinite
        Returns:
          a new Spectrum object
        """

        if not isinstance(scaling_factor, UFloat):
            try:
                scaling_factor = float(scaling_factor)
            except (TypeError, ValueError):
                raise TypeError("Spectrum must be multiplied/divided by a scalar")
            if (
                    scaling_factor == 0
                    or numpy.isinf(scaling_factor)
                    or numpy .isnan(scaling_factor)
            ):
                raise ValueError("Scaling factor must be nonzero and finite")
        else:
            if (
                    scaling_factor.nominal_value == 0
                    or numpy.isinf(scaling_factor.nominal_value)
                    or numpy.isnan(scaling_factor.nominal_value)
            ):
                raise ValueError("Scaling factor must be nonzero and finite")
        if div:
            multiplier = 1 / scaling_factor
        else:
            multiplier = scaling_factor

        if self._counts is not None:
            data_arg = {"counts": self.counts * multiplier}
        else:
            data_arg = {"cps": self.cps * multiplier}

        if self.is_calibrated:
            spect_obj = Spectr(bin_edges_kev=self.bin_edges_kev, **data_arg)
        else:
            spect_obj = Spectr(bin_edges_raw=self.bin_edges_raw, **data_arg)
        return spect_obj

    def parse_xmode(self, xmode):
        """Parse the x-axis mode to get the associated data and plot label.
        Parameters
        ----------
        xmode : {'energy', 'channel'}
            Mode (effectively units) of the x-axis
        Returns
        -------
        xedges, xlabel
            X-axis bin edges and a suitable label for plotting
        Raises
        ------
        ValueError
            If the xmode parameter is unsupported
        """
        if xmode == "energy":
            xedges = self.bin_edges_kev
            xlabel = "Energy [keV]"
        elif xmode == "channel":
            xedges = self.bin_edges_raw
            xlabel = "Channel"
        else:
            raise ValueError(f"Неподдерживаемый xmode: {xmode:s}")
        return xedges, xlabel

    def parse_ymode(self, ymode):
        """Parse the y-axis mode to get the associated data and plot label.
        Parameters
        ----------
        ymode : {'counts', 'cps', 'cpskev'}
            Mode (effectively units) of the y-axis
        Returns
        -------
        ydata, yuncs, ylabel
            Y-axis data, uncertainties, and a suitable label for plotting
        Raises
        ------
        ValueError
            If the ymode parameter is unsupported
        """
        if ymode == "counts":
            ydata = self.counts_vals
            yuncs = self.counts_uncs
            ylabel = "Counts"
        elif ymode == "cps":
            ydata = self.cps_vals
            yuncs = self.cps_uncs
            ylabel = "cps [1/s]"
        elif ymode == "cpskev":
            ydata = self.cpskev_vals
            yuncs = self.cpskev_uncs
            ylabel = "cps [1/s/keV]"
        else:
            raise ValueError(f"Unsupported ymode: {ymode:s}")
        return ydata, yuncs, ylabel



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

    def fill_between(self, **kwargs):
        """Plot a spectrum with matplotlib's fill_between command
        Args:
          xmode:  define what is plotted on x axis ('energy' or 'channel'),
                  defaults to energy if available
          ymode:  define what is plotted on y axis ('counts', 'cps', 'cpskev'),
                  defaults to counts
          xlim:   set x axes limits, if set to 'default' use special scales
          ylim:   set y axes limits, if set to 'default' use special scales
          ax:     matplotlib axes object, if not provided one is created
          yscale: matplotlib scale: 'linear', 'log', 'logit', 'symlog'
          title:  costum plot title
          xlabel: costum xlabel value
          ylabel: costum ylabel value
          kwargs: arguments that are directly passed to matplotlib's
                  fill_between command. In addition it is possible to pass
                  linthresh if ylim='default' and ymode='symlog'.
        Returns:
          matplotlib axes object
        """

        plotter = plotting.PlotSpectrum(self, **kwargs)
        return plotter.fill_between()