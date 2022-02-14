'''
Чтение спектр-фала с разрешением .spe
'''
import datetime
import os  # Модуль, позволяющий работать с операционной системой
import warnings
import numpy
from ATOMTEXSPECTR.encoding import UniDet
import dateutil.parser
from parsers import ReadingParserError, ReadingParserWarning


def reading(filename, debbuging = False):
    '''
    Parse the ASCII .ats file and return a dictionary of data
    :param filename:Имя файла (строковый тип данных). The filename of the CNF file to read.
    :param debbuging: bool
    Следует ли выводить отладочную (debugging) информацию. По умолчанию False.
    :return: Dictionary of data in spectrum
    '''

    print('Читаемый спектр-файл:' + filename)  # Вывод названия файла
    namefile, extension = os.path.splitext(filename)
    # Проверка кодировки файла
    encoding = UniDet.encodingfile(path)
    # Проверка соответствия формату файла
    if extension != '.spe':
        raise ReadingParserError('Формат файла неверный' + extension)

    # Инициализируемый словарь для заполнения данными спектра по ходу парсинга файла
    data = dict()
    # Парсинг файла
    DATESTART = None
    COUNTS = list()
    CHANNELS = list()
    COEFFICIENTS = list()
    MCA_166_ID = list()
    ENERGY = list()
    # ENER_FIT = list()
    RADIONUCLIDES = list()
    GPS = dict()
    SIGMA = list()

    with open(filename, encoding = encoding) as file:

        # Список с компонентами из строк файла
        LINES = [LINE.strip() for LINE in file.readlines()]

        for item in range(len(LINES)):
            if LINES[item] == '$MCA_166_ID:':
                item = item + 1
                MCA_166_ID.append(item)

            elif LINES[item] == "$DATE_MEA:":
                item += 1
                DATESTART = dateutil.parser.parse(LINES[item])
                if debbuging:
                    print(DATESTART)

            elif LINES[item] == "$DATA:":
                item += 1
                FIRSTCHANNEL = float(LINES[item].split(" ")[0])
                if FIRSTCHANNEL != 0:
                    raise ReadingParserError(f"First channel is not 0: {FIRSTCHANNEL}")
                NUMBERCHANNEL = float(LINES[item].split(" ")[1])
                if debbuging:
                    print(FIRSTCHANNEL, NUMBERCHANNEL)
                j = FIRSTCHANNEL
                while j <= FIRSTCHANNEL + NUMBERCHANNEL:
                    item += 1
                    COUNTS = numpy.append(COUNTS, int(LINES[item]))
                    CHANNELS = numpy.append(CHANNELS, j)
                    j += 1

            elif LINES[item] == '$ENER_FIT:':
                item += 1
                ENER_FIT = LINES[item].split(" ")
                ENERGYFIT = [float(index) for index in ENER_FIT]

            elif LINES[item] == '$ROI:':
                item += 1
                ROI = LINES[item]

            elif LINES[item] == '$ENER_TABLE:':
                item += 1
                LENCHANNEL = int(LINES[item].split(" ")[0])
                FIRSTCHANNEL = float(LINES[item + 1].split(" ")[0])
                j = FIRSTCHANNEL
                while j < LENCHANNEL:
                    item += 1
                    ENERGY = numpy.append(ENERGY, float(LINES[item].split(" ")[1]))
                    j += 1

            elif LINES[item] == '$SIGM_DATA:':
                item += 1
                LENCHANNEL = int(LINES[item].split(" ")[0])
                FIRSTCHANNEL = float(LINES[item + 1].split(" ")[0])
                j = FIRSTCHANNEL
                while j < LENCHANNEL:
                    item += 1
                    SIGMA = numpy.append(SIGMA, float(LINES[item].split(" ")[1]))
                    j += 1

            elif LINES[item] == '$TEMPERATURE:':
                TEMP = float(LINES[item + 1])

            elif LINES[item] == '$SCALE_MODE:':
                SCALE = float(item + 1)

            elif LINES[item] == '$DOSE_RATE:':
                DOSERATE = float(LINES[item + 1])
                if DOSERATE == None or 0:
                    ReadingParserWarning('Ошибка получения значения мощности дозы:' + str(DOSERATE))

            elif LINES[item] == '$DU_NAME:':
                DUNAME = LINES[item + 1]

            # Идентифицированные радионуклиды в ходе измерений
            elif LINES[item] == '$RADIONUCLIDES:':
                RADIONUCLIDES.append(LINES[item + 1].split(" "))

            elif LINES[item] == '$ACTIVITYRESULT:':
                item += 1
                ACTIVITY = LINES[item]
                if debbuging and len(ACTIVITY) == 0:
                    print('Ошибка получения информации об активности, нет данных.')
                # Значение эффективноси регистрации

            elif LINES[item] == '$EFFECTIVEACTIVITYRESULT:':
                EFFECTIVEACTIVITYRESULT = LINES[item + 1]
                if len(EFFECTIVEACTIVITYRESULT) == 0 and debbuging:
                    print('Ошибка получения информации об эффективности регистрации, нет данных.')

            # Код геометрии
            elif LINES[item] == '$GEOMETRY:':
                GEOMETRY = LINES[item + 1]
                if len(GEOMETRY) == 0 and debbuging:
                    print('Ошибка получения информации о геометрии, нет данных.')
                    # Дата производства блока детектирования, спектрометра или любого друго устройства, позволяющего
                    # записывать спектр в формате .spe

            # Переменная, содержащая дату прозводства
            #           elif LINES[item] == '$DATE_MANUFACT:':
            #               DATAMANUF = dateutil.parser.parse(LINES[item + 1])

            elif LINES[item] == '$GAIN:':
                gain = LINES[item + 1]

            # GPS data
            elif LINES[item] == '$GPS:':
                Longitude = LINES[item + 1].split(" ")[1]
                Latitude = LINES[item + 2].split(" ")[1]
                Alternate = LINES[item + 3].split(" ")[1]
                Speed = LINES[item + 4].split(" ")[1]
                Direction = LINES[item + 5].split(" ")[1]
                Valid = LINES[item + 6].split(" ")[1]
                GPS['Longitude'] = Longitude
                GPS['Latitude'] = Latitude
                GPS['Alternate'] = Alternate
                GPS['Speed'] = Speed
                GPS['Direction'] = Direction
                GPS['Valid'] = Valid

            elif LINES[item].startswith("$"):
                key = LINES[item][1:].rstrip(":")
                item += 1
                values = []
                while item < len(LINES) and not LINES[item].startswith("$"):
                    values.append(LINES[item])
                    item += 1
                if item < len(LINES):
                    if LINES[item].startswith("$"):
                        item -= 1
                if len(values) == 1:
                    values = values[0]
                data[key] = values
            # else:
            #     warnings.warn(f'Строка {item + 1} неизвестная: ' + LINES[item], ReadingParserWarning)
            item += 1

    data["Counts"] = COUNTS
    data["Time"] = DATESTART
    data['GPS'] = GPS
    data['temp'] = TEMP
    data['gain'] = gain
    data['energy'] = ENERGY
    # # data[]

    # Калибровка

    return data

if __name__ == '__main__':
    path = r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe'
    # encoding = UniDet.encodingfile(path)
    file = reading(path)
    # print(file.keys())
    # print(file['energy'])
    print(file)
