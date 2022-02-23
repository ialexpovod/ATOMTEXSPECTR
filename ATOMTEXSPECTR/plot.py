'''
Инструменты для плоттинга (графического представления) спектров
'''
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy
from ATOMTEXSPECTR.warn import plot_error
import warnings

class plot_spectrum:  # PlotSpectrum(object) без родительского класса. Не наследуется ни начем
    '''
    Класс для обработки спектров.
    '''

    def __init__(self, spectrum, *args, **kwargs):
        # Добавим атрибуты методу, конструктуру__init__
        # Это не функция, а метод. Функция меняет свое название на метод когда находится внутри класса.
        # Каждый метод должен иметь хотя бы один аргумент.
        # оператор "звездочка" позволяет «распаковывать» объекты, внутри которых хранятся некие элементы.
        # a = [1,2,3]
        # b = [*a,4,5,6]
        # print(b) # [1,2,3,4,5,6]
        # def printScores(student, *scores):
        #    print(f"Student Name: {student}")
        #    for score in scores:
        #       print(score)
        # printScores("Jonathan",100, 95, 88, 92, 99)
        # """
        # Student Name: Jonathan
        # 100
        # 95
        # 88
        # 92
        # 99
        # """
        # def printPetNames(owner, **pets):
        #    print(f"Owner Name: {owner}")
        #    for pet,name in pets.items():
        #       print(f"{pet}: {name}")
        # printPetNames("Jonathan", dog="Brock", fish=["Larry", "Curly", "Moe"], turtle="Shelldon")
        # """
        # Owner Name: Jonathan
        # dog: Brock
        # fish: ['Larry', 'Curly', 'Moe']
        # turtle: Shelldon
        # """
        #
        '''
        Args:
            :param  spectrum:    Пример представления спектра в графическом виде (Плоттинг спектра)
            :param  args:   (Позиционный аргумент) Matplotlib как строковый тип данных для плоттинга
            :param  kwargs:      Аргументы, которые напрямую передаются команде построения графика matplotlib.
                                Кроме того, можно передать linthresh, если ylim='по дефолту' и ymode='symbolelog'
                    xmode:  дефайн на оси абсцисс ('energy' или 'channel')
                    ymode:  дефайн на оси ординат ('counts', 'cps', 'cpskev')
                    xlim:   пределы оси абсцисс
                    ylim:   пределы оси ординат
                    ax:     объект осей matplotlib, если он не предусмотрен, создается с его помощью
                    yscale: matplotlib вид 'linear', 'log', 'logit', 'symlog'
                    title:  кастомное название спектра, по дефолту -- название файла если есть
                    xlabel: кастомное название оси абсцисс
                    ylabel: кастомное название оси ординат
        '''

        self._xedges = None
        self._ydata = None
        self._xmode = None
        self._ymode = None
        self._xlabel = None
        self._ylabel = None
        self._ax = None
        self._xlim = None
        self._ylim = None
        self._linthresh = None
        self.spectrum = spectrum

        # Проверяем наличие объектов при помощи hasttr()
        # class MyObj:
        #     name = 'Chuck Norris'
        #     phone = '+666111000'
        #     country = 'Norway'
        # # Проверим наличие атрибутов в объекте MyObj:
        # x = hasattr(MyObj, 'phone')
        # y = hasattr(MyObj, 'age')
        # print(x, y)
        # # Вывод
        # True, False
        if hasattr(args, "__len__") and len(args) in [0, 1]:
            self.args = args
        else:
            raise plot_error("Неправильное количество позиционных аргументов!")
        # d = {'a' :1, 'c' :2}
        # print(d.get('b', 0)) # return 0
        # print(d.get('c', 0)) # return 2
        # d = {'a' :1, 'c' :2}
        # print(d.pop('c', 0)) # return 2
        # print(d) # returns {'a': 1}
        # print(d.get('c', 0)) # return 0
        self.xmode = kwargs.pop("xmode", None)
        self.ymode = kwargs.pop("ymode", None)
        self.xlim = kwargs.pop("xlim", None)
        self.ylim = kwargs.pop("ylim", None)
        self.ax = kwargs.pop("ax", None)
        self._linthresh = kwargs.pop("linthresh", None)

        if "linthreshy" in kwargs:
            warnings.warn(
                "linthreshy устарел, вместо этого используйте linthresh!",
                DeprecationWarning,
            )
            self._linthresh = kwargs.pop("linthreshy")
        self.yscale = kwargs.pop("yscale", None)
        self.title = kwargs.pop("title", None)
        self.xlabel = kwargs.pop("xlabel", None)
        self.ylabel = kwargs.pop("ylabel", None)

        self.kwargs = kwargs

    # def makebold(fn):
    #     def wrapped():
    #         return "<b>" + fn() + "</b>"
    #     return wrapped
    #
    # def makeitalic(fn):
    #     def wrapped():
    #         return "<i>" + fn() + "</i>"
    #     return wrapped
    #
    # @makebold
    # @makeitalic
    # def hello():
    #     return "hello habr"
    # print hello() ## выведет <b><i>hello habr</i></b>
    @property  # декоратор
    def xmode(self):
        """
        Возвращает имеющщийся способ построения спектра по оси абсцисс.
        """
        return self._xmode

    @xmode.setter
    def xmode(self, mode):
        '''
        Метод определения данных по оси абсцисс. Также дефайн данных edges и xlabel.
        :param mode: energy (or kev, mev, e), channel (or channels, chn, chns, c, K)

        '''
        # Вначале, выбор _xmode
        # Если спектр не калиброван, если в методе не указана какая ось,
        # по дефолту -- ось абсцисс - энергия, если спектр калиброван. Иначе - каналы.
        if mode is None:
            if self.spectrum.is_calibrated_for_energy:  # Если есть калибровка по энергии, то
                self._xmode = "energy"
            else:
                self._xmode = "channel"
        else:
            if mode.lower() in ("kev", "energy", "mev", "e"):
                if not self.spectrum.is_calibrated_for_energy:  # Если False (нет калибровок)...
                    raise plot_error(  # Исключение
                        "Спектр не калиброван, как же так?!"
                        "Ось абсцисс была объявлена как энергия "
                    )
                self._xmode = "energy"
            elif mode.lower() in ("channel", "channels", "chn", "chns", "c", "k"):
                self._xmode = "channel"
            else:
                raise plot_error(f"Неизвестный формат данных для оси: {mode}")

        # Тогда, зададим _xedges и _xlabel соновываясь на _xmode
        xedges, xlabel = self.spectrum.parsing_abscissa(self._xmode)
        self._xedges = xedges
        # Название осей
        possible_labels = ["Energy [keV]", "Channel"]
        if self._xlabel in possible_labels or self._xlabel is None:
            # Only reset _xlabel if it's an old result from parse_xmode or None
            self._xlabel = xlabel

    @property
    def ymode(self):
        '''
            Возвращает имеющщийся способ построения спектра по оси ординат.
            :param self:
            :return: self._ymade
        '''
        return self._ymode

    @ymode.setter
    def ymode(self, mode):
        """
            Определение способа задания данных по оси ординат. Также дефайн ydata и ylabel
            Не проверяет определена ли скорость счета (cps). Если таковой нет, это приведет
            к SpectrumERROR()
            :param  mode: counts, cps
        """

        # Вначале, зададим _ymode
        # Если способу задания оси абсцисс не присвоено ничего...
        if mode is None:
            #
            if self.spectrum._counts is not None:  # Если есть, содержит
                self._ymode = "counts"  # Присвоить
            else:
                self._ymode = "cps"
        elif mode.lower() in ("count", "counts", "cnt", "cnts"):
            if self.spectrum._counts is None:
                raise plot_error("У спектра есть отсчеты в каналах, которые не заданы.")
            self._ymode = "counts"
        elif mode.lower() == "cps":
            self._ymode = "cps"
        elif mode.lower() == "cpskev":
            self._ymode = "cpskev"
        else:
            raise plot_error(f"Unknown y data mode: {mode}")

        # Then, set the _ydata and _ylabel based on the _ymode
        ydata, _, ylabel = self.spectrum.parsing_ordinate(self._ymode)
        self._ydata = ydata
        possible_labels = ["Counts", "Countrate [1/second]", "Countrate [1/second/keV]"]
        if self._ylabel in possible_labels or self._ylabel is None:
            # Only reset _ylabel if it's an old result from parse_ymode or None
            self._ylabel = ylabel

    @property
    def ax(self):
        """
            Возвращает имеющийся matplotlib axes объект используемый для графического построения спектра.
            Если объект axes не определен, то метод ax() его создаст
        """
        if self._ax is None:
            _, self._ax = plt.subplots()
        return self._ax

    @ax.setter
    def ax(self, ax):
        """
            Определяет существующий matplotlib axes объект для построения спектра.
            :param ax: задание axes
        """
        self._ax = ax

    @property
    def xlabel(self):
        """
            Возвращает имеющийся лэйбл оси абсцисс.
        """
        return self._xlabel

    @xlabel.setter
    def xlabel(self, label):
        """
            Задание xlabel своим (кастомным) значением.
        """
        if label is not None:
            self._xlabel = label

    @property
    def ylabel(self):
        """
            Возвращает имеющийся лэйбл оси ординат.
        """

        return self._ylabel

    @ylabel.setter
    def ylabel(self, label):
        """
            Задание ylabel своим (кастомным) значением.
        """
        if label is not None:
            self._ylabel = label

    @property
    def yerror(self):
        """
            Возвращает массив статистических ошибок для заданного способа определения оси ординат.
        """
        if self._ymode == "counts":
            return self.spectrum.counts_uncs
        elif self._ymode == "cps":
            return self.spectrum.cps_uncs
        elif self._ymode == "cpskev":
            return self.spectrum.cpskev_uncs

    def get_corners(self):
        """
        Создает пошаговую версию текущих данных о спектре.
        Return:
        :return  xcorner, ycorner: x и y значения, которые могут использоваться напрямую в построении спектра.
        """
        return self.bin_edges_and_heights_to_steps(
            self._xedges, unumpy.nominal_values(self._ydata)
        )

    def _prepare_plot(self, **kwargs):
        """
        Подготовка к построению спектра.
        :param kwargs:
        :return: пошаговую версию данных о спектре
        """

        self.kwargs.update(**kwargs)
        if not self.ax.get_xlabel():
            self.ax.set_xlabel(self._xlabel)
        if not self.ax.get_ylabel():
            self.ax.set_ylabel(self._ylabel)
        if self.yscale is not None:
            self.ax.set_yscale(self.yscale)
        if self.title is not None:
            self.ax.set_title(self.title)
        elif "filename" in self.spectrum.data:
            self.ax.set_title(self.spectrum.data["filename"])
        if self._xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self._ylim is not None:
            self.ax.set_ylim(self.ylim)
            if self.yscale == "symlog" and self._ylim == "default":
                self.ax.set_yscale(self.yscale, linthresh=self.linthresh)
        return self.get_corners()

    def plot(self, *args, **kwargs):
        """
            Создает актуальный спектр по методу plot с помощью модуля matplotlib.

            :param args:        Matplotlib plot like format string
            :param kwargs:      Any matplotlib plot() keyword argument, overwrites
                                previously defined keywords
        """

        # Проверка: есть ли метод __len__ в объекте (набор элементов) args и есть ли в нем что (> 0)
        if hasattr(args, "__len__") and len(args) > 0:
            # Тогда присвоим с
            # Классам нужен способ чтобы ссылаться на свимх же себя. Это способ сообщения между экземплярами.
            # Пример: x = 'text', 'text' - объект-строка (класс), а x - экземпляр. Тогда x задат несколько методов, которые и сами ссылаются друг на друга
            self.args = args

        if not hasattr(self.args, "__len__") or not len(self.args) in [0, 1]:
            raise plot_error("Неверное число позиционных аргументов!")

        xcorners, ycorners = self._prepare_plot(**kwargs)
        self.ax.plot(xcorners, ycorners, *self.args, **self.kwargs)
        return self.ax

    def fill_between(self, **kwargs):
        """
            Создадим фактическое отображение спектра
            с помощью matplotlib's fill_between метода.

            :param  kwargs:     Any matplotlib fill_between() keyword argument, overwrites
                                previously defined keywords
        """

        xcorners, ycorners = self._prepare_plot(**kwargs)
        self.ax.fill_between(xcorners, ycorners, **self.kwargs)
        return self.ax

    def errorbar(self, **kwargs):
        """
            Создадим погрешности (inaccuracy) с помощью matplotlib's errorbar метода в plot.

        :param  kwargs:     Любой matplotlib errorbar() ключевой аргумент,
                            перезаписывает ранее определенные ключевые слова
        """

        self._prepare_plot(**kwargs)
        xdata = (self._xedges[0:-1] + self._xedges[1:]) * 0.5

        if "fmt" in self.kwargs:
            self.fmt = self.kwargs.pop("fmt")

        if hasattr(self.fmt, "__len__") and len(self.fmt) == 0:
            self.fmt = (",",)

        if not hasattr(self.fmt, "__len__") or len(self.fmt) != 1:
            raise plot_error("Неверное число заданных аргументов args")

        self.ax.errorbar(
            xdata, self._ydata, yerr=self.yerror, args=self.args[0], **self.kwargs
        )

    def errorband(self, **kwargs):
        """
        Create an errorband with matplotlib's plot fill_between method.
        Args:
          kwargs: Any matplotlib fill_between() keyword argument, overwrites
                  previously defined keywords
        """

        self._prepare_plot(**kwargs)

        alpha = self.kwargs.pop("alpha", 0.5)

        xcorners, ycorlow = self.bin_edges_and_heights_to_steps(
            self._xedges, unumpy.nominal_values(self._ydata - self.yerror)
        )
        _, ycorhig = self.bin_edges_and_heights_to_steps(
            self._xedges, unumpy.nominal_values(self._ydata + self.yerror)
        )
        self.ax.fill_between(xcorners, ycorlow, ycorhig, alpha=alpha, **self.kwargs)
        return self.ax


    @staticmethod
    def bin_edges_and_heights_to_steps(bin_edges, heights):
        """
            Алтернатива matplotlib's drawstyle='steps'
        """

        # print(bin_edges, heights)
        assert len(bin_edges) == len(heights) + 1
        x = np.zeros(len(bin_edges) * 2)
        y = np.zeros_like(x)  # Функция zeros_like() возвращает новый массив из нулей с формой и типом данных указанного массива x.
        x[::2] = bin_edges.astype(float)
        x[1::2] = bin_edges.astype(float)
        y[1:-1:2] = heights.astype(float)
        y[2:-1:2] = heights.astype(float)
        return x, y


    @staticmethod
    def dynamic_min(data_min, min_delta_y):
        '''
        Метод возвращает нижний предел, граничное значение для оси ординат.
        на основе значения данных. Нижний предел - следующая степень 10-ая,
        или 3 * в степени 10, ниже минимума.

        :param data_min: минимум входных данных (может быть как int(), так и float())
        :param min_delta_y: минимальный шаг по оси ординат
        '''
        if data_min > 0:
            ceil10 = 10 ** (np.ceil(np.log10(data_min)))  # Функция ceil() округляет к большему целому числу.
            # np.ceil([-1.5, 1.5])
            # >> array([-1, 2])
            sig_fig = np.floor(10 * data_min / ceil10) # функция floor() округляет к меньшему числу целому
            # np.floor([-1.5, 1.5])
            # >> array([-2, 1])
            # np.floor([ - 1.6, 1.6])
            # >> array([ - 2, 1])
            if sig_fig <= 3:
                ymin = ceil10 / 10
            else:
                ymin = ceil10 / 10 * 3
        elif data_min == 0:
            ymin = min_delta_y / 10.0
        else:
            # когда data_min < 0 - negative
            floor10 = 10 ** (np.floor(np.log10( - data_min)))
            sig_fig = np.floor( - data_min / floor10)
            if sig_fig < 3:
                ymin = - floor10 * 3
            else:
                ymin = -floor10 * 10
        return ymin

    @staticmethod
    def dynamic_max(data_max, yscale):
        '''
            Метод позволяет получить верхний предел оси ординат на основе вхожных значений.
            Верхний предел - степень 10 или 3 * степень выше макс. значения.

            :param data_max:
            :param yscale:
            :return:
        '''
        floor10 = 100 ** (np.floor(np.log10(data_max)))
        sig_fig = np.ceil(data_max / floor10)
        if yscale == 'linear':
            sig_fig = np.floor(data_max / floor10)
            ymax = floor10 * (sig_fig + 1)
        elif sig_fig < 3:
            ymax = floor10 * 3
        else:
            ymax = floor10 * 10

        return np.maximum(ymax, 0)

    @property
    def xlim(self):
        '''
            При отображении графиков (спектра граничные значения по каждой из осей по умолчанию определяются
            автоматичеки, исходя из набора входных данных. Иногда требуется указывать
            свои граничные значения.

            Метод возвращает гоаничные значения для оси абсцисс.
        :return:
        '''

        if self._xlim is None or self._xlim == 'default':
            return np.min(self._xedges), np.max(self._xedges)

        return self._xlim

    @xlim.setter
    def xlim(self, limits):
        '''
        Возврат множества граничных значений для оси абсцисс.
        :param limits:
        :return:
        '''
        if (
            limits is not None and limits != 'default' and (not hasattr(limits, "__len__") or len(limits) != 2)
        ):
            raise plot_error(f'xlim должна быть размером 2: {limits}')
        self._xlim = limits

    @property
    def ylim(self):
        '''
            Метод возвращет заданные граничные значения ylim,
            в зависимости yscale, ydata.
        :return:
        '''
        if self._ylim is None or self._ylim == 'default':
            yscale = self.yscale
            if yscale is None:
                yscale = self.ax.get_yscale()
            min_ind = np.argmin(np.abs(self._ydata[self._ydata != 0]))
            delta_y = np.abs(self._ydata - self._ydata[min_ind])
            min_delta_y = np.min(delta_y[delta_y > 0])

            data_min = np.min(self._ydata)
            if yscale == 'linear':
                ymin = 0
            elif yscale == 'log' and data_min < 0:
                raise plot_error('Нельзя отобразить отрицательные значения в масштабе Log; '
                                    'используйте symlog scale')
            elif yscale == 'symlog' and data_min >= 0:
                ymin = 0
            else:
                ymin = self.dynamic_min(data_min, min_delta_y)

            data_max = np.max(self._ydata)
            ymax = self.dynamic_max(data_max, yscale)
            return ymin, ymax
        return self._ylim

    @ylim.setter
    def ylim(self, limits):
        '''
            Задает граничные значения для оси ординат (ylim).
        :param limits:
        :return:
        '''
        if (
            limits is not None
            and limits != 'default' and
                (not hasattr(limits, '__len__') or len(limits) != 2)
        ):
            raise  plot_error(f'ylim должны быть размером в 2: {limits}')
        self._ylim = limits

    @property
    def linthresh(self):
        '''
        Возвращает linthresh, с условием наличия ydata.
        :return:
        '''
        if self._linthresh is not None:
            return self._linthresh
        min_ind = np.argmin(self._ydata[self._ydata != 0])           # озвращает индекс минимального значения указанной оси
        delta_y = np.abs(self._ydata - self._ydata[min_ind])
        return np.min(delta_y[delta_y > 0])

    @property
    def linthreshy(self):
        warnings.warn("linthreshy устарел, вместо этого используйте linthresh", DeprecationWarning)
        return self.linthresh

