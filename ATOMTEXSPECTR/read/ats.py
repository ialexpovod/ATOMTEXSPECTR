'''
Чтение спектр-файла с разрешением .ats
'''
import os
import numpy
from ATOMTEXSPECTR.encoding import UniDet
import dateutil.parser
from parsers import ReadingParserError, ReadingParserWarning

def reading(filename, debugging = False):
    '''
    Parse the ASCII .ats file and return a dictionary of data
    :param filename:        Имя файла (строковый тип данных). The filename of the CNF file to read.
    :param debbuging:       bool
                            Следует ли выводить отладочную (debugging) информацию. По умолчанию False.
    :return:                Dictionary of data in spectrum
    '''
    print('Читаемый спектр-файл:' + filename)  # Вывод названия файла
    namefile, extension = os.path.splitext(filename)
    # Проверка соответствия формату файла
    if extension != '.ats':
        raise ReadingParserError('Формат файла неверный' + extension)
    # Инициализируемый словарь для заполнения данными спектра по ходу парсинга файла
    # Проверка кодировки файла
    encoding = UniDet.encodingfile(filename)
    data = dict()
    # Парсинг файла
    DATESTART = None
    DATE_MANUFACT = None
    Time = None
    CHANNELS = list()
    COEFFICIENTS = list()
    MCA_166_ID = list()
    energy = list()
    resultion = list()
    counts = list()
    RADIONUCLIDES = list()
    GPS = dict()
    SIGMA = list()
    with open(filename, encoding = encoding) as file:
        # Список с компонентами из строк файла
        LINES = [LINE.strip().split(" ") for LINE in file.readlines()]
        # print()
        # for item in range(len(LINES)):
        #    print()
        # print(LINES)
        for item in range(len(LINES)):
            if LINES[item][0] == "TIME":
                Time = int(LINES[item][2])
                if debugging:
                    print('Время набора спектра = ', Time)
            elif LINES[item][0] == 'ACTIVITY':
                Activity = float(LINES[item][2])
            elif LINES[item][0] == 'REMARK':
                if len(LINES[item]) >= 3:
                    remark = LINES[item][2]
                else:
                    remark = None
            elif LINES[item][0] == 'GEOMETRY':
                if len(LINES[item]) >= 3:
                    geometry = LINES[item][2]
                else:
                    geometry = None
            elif LINES[item][0] == 'WEIGHT':
                if len(LINES[item]) >= 3:
                    weight = float(LINES[item][2])
            elif LINES[item][0] == 'DATE':
                # Преобразование списка в строку методом join
                Date = dateutil.parser.parse(" ".join(LINES[item][2:]))

            elif LINES[item][0] == 'RADIATION':
                if len(LINES[item]) >= 3:
                    radiation = LINES[item][2]

           # ----------------GPS ----------------
            elif LINES[item][0] == 'GPS_LAT':
                if len(LINES[item]) >= 3:
                    GPS_LAT = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_LON':
                if len(LINES[item]) >= 3:
                    GPS_LON = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_ALT':
                if len(LINES[item]) >= 3:
                    GPS_ALT = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_SPD':
                if len(LINES[item]) >= 3:
                    GPS_SPD = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_DIR':
                if len(LINES[item]) >= 3:
                    GPS_DIR = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_VLD':
                if len(LINES[item]) >= 3:
                    GPS_VLD = float(LINES[item][2])


            elif LINES[item][0] == 'DOSERATE':
                if len(LINES[item]) >= 3:
                    Dose = float(LINES[item][2])
            elif LINES[item][0] == 'DU_TYPE':
                if len(LINES[item]) >= 3:
                    DU = LINES[item][2]
            elif LINES[item][0] == 'SCALE_MODE':
                if len(LINES[item]) >= 3:
                    SCALE_MODE = float(LINES[item][2])
            elif LINES[item][0] == 'SERIAL':
                if len(LINES[item]) >= 3:
                    serial = int(LINES[item][2])
            elif LINES[item][0] == 'FIRMWARE':
                if len(LINES[item]) >= 3:
                    firmware = float(LINES[item][2])
            elif LINES[item][0] == 'HARDWARE':
                if len(LINES[item]) >= 3:
                    harware = float(LINES[item][2])
            elif LINES[item][0] == 'DISTANCE':
                if len(LINES[item]) >= 3:
                    distance = float(LINES[item][2])
            elif LINES[item][0] == 'SWIDTH':
                if len(LINES[item]) >= 3:
                    SWIDTH = float(LINES[item][2])
            elif LINES[item][0] == 'CPSOUTOFRANGE':
                if len(LINES[item]) >= 3:
                    CPSOUTOFRANGE = float(LINES[item][2])
            elif LINES[item][0] == 'DATE_MANUFACT':
                if len(LINES[item]) >= 3 and len(LINES[item][2]) == 7:
                    DATE_MANUFACT = dateutil.parser.parse(LINES[item][2]).strftime("%Y-%m")
                else:
                    raise ReadingParserError(f"Неизвестное значение даты: {LINES[item][2]}")

            # realtime
            elif LINES[item][0] == 'REALTIME':
                if len(LINES[item]) >= 3:
                    realtime = int(LINES[item][2])
            elif LINES[item][0] == 'CPS':
                if len(LINES[item]) >= 3:
                    cps = float(LINES[item][2])
            elif LINES[item][0] == 'ACTIVITYRESULT':
                if len(LINES[item]) >= 3:
                    ACTIVITYRESULT = float(LINES[item][2])
                else:
                    ACTIVITYRESULT = None
            elif LINES[item][0] == 'EFFECTIVEACTIVITYRESULT':
                if len(LINES[item]) >= 3:
                    EFFECTIVEACTIVITYRESULT = LINES[item][2]
                else:
                    EFFECTIVEACTIVITYRESULT = None
            elif LINES[item][0] == 'MIX':
                if len(LINES[item]) >= 3:
                    MIX = LINES[item][2]
                else:
                    MIX = None
            elif LINES[item][0] == 'SPECTRUMPROCESSED':
                if len(LINES[item]) >= 3:
                    SPECTRUMPROCESSED = float(LINES[item][2])

            elif LINES[item][0] == 'BGNDSUBTRACTED':
                if len(LINES[item]) >= 3:
                    BGNDSUBTRACTED = float(LINES[item][2])
            elif LINES[item][0] == 'ENCRYPTED':
                if len(LINES[item]) >= 3:
                    ENCRYPTED = float(LINES[item][2])
            elif LINES[item][0] == 'NEUTRON_CPS':
                if len(LINES[item]) >= 3:
                    NEUTRON_CPS = float(LINES[item][2])
            elif LINES[item][0] == 'NEUTRON_COUNT':
                if len(LINES[item]) >= 3:
                    NEUTRON_COUNT = int(LINES[item][2])
            elif LINES[item][0] == 'NEUTRON_DOSERATE':
                if len(LINES[item]) >= 3:
                    NEUTRON_DOSERATE = float(LINES[item][2])
            elif LINES[item][0] == 'STATUS_OF_HEALTH':
                if len(LINES[item]) >= 3:
                    STATUS_OF_HEALTH = LINES[item][2]
            # GPS
            elif LINES[item][0] == 'GPS_RMCHOUR':
                if len(LINES[item]) >= 3:
                    GPS_RMCHOUR = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_RMCMINUTE':
                if len(LINES[item]) >= 3:
                    GPS_RMCMINUTE = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_RMCMINUTE':
                if len(LINES[item]) >= 3:
                    GPS_RMCMINUTE = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_RMCSECOND':
                if len(LINES[item]) >= 3:
                    GPS_RMCSECOND = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_RMCDAY':
                if len(LINES[item]) >= 3:
                    GPS_RMCDAY = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_RMCMONTH':
                if len(LINES[item]) >= 3:
                    GPS_RMCMONTH = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_RMCYEAR':
                if len(LINES[item]) >= 3:
                    GPS_RMCYEAR = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_GSAPDOP':
                if len(LINES[item]) >= 3:
                    GPS_GSAPDOP = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_GSAHDOP':
                if len(LINES[item]) >= 3:
                    GPS_GSAHDOP = float(LINES[item][2])
            elif LINES[item][0] == 'GPS_GSAVDOP':
                if len(LINES[item]) >= 3:
                    GPS_GSAVDOP = float(LINES[item][2])

            elif LINES[item][0] == 'RADIONUCLIDES':
                radionuclides = LINES[item][2:]
            elif LINES[item][0] == 'TEMPERATURE':
                temp = float(LINES[item][2])

            elif LINES[item][0] == 'ECALIBRATION':
                ecal = int(LINES[item][2])
                if ecal != 0 or ecal is not None:
                    j = 0
                    while j <= ecal- 1:
                        item += 1
                        # print(LINES[item][0])
                        energy = numpy.append(energy, float(LINES[item][0]))
                        j += 1
                else:
                    raise ReadingParserError(f'Ошибка получения данных по энергетической калибровке: {ecal}')

            elif LINES[item][0] == 'RCALIBRATION':
                res = int(LINES[item][2])
                if res != 0 or res is not None:
                    j = 0
                    while j <= res- 1:
                        item += 1
                        resultion = numpy.append(resultion, float(LINES[item][0]))
                        j += 1
                else:
                    raise ReadingParserError(f'Ошибка получения данных калибровки по разрешению: {ecal}')

            elif LINES[item][0] == 'SPECTR':
                spectr = int(LINES[item][2])
                if spectr != 0 or spectr is not None:
                    j = 0
                    while j <= spectr- 1:
                        item += 1
                        counts = numpy.append(counts, float(LINES[item][0]))
                        j += 1
                else:
                    raise ReadingParserError(f'Ошибка получения данных по отсчетам: {spectr}')

            elif LINES[item][0] == '$DR_ERR:':
                if len(LINES[item]) >= 3:
                    split = LINES[item][2]
                    if "," in split:
                        DR_ERR = float(".".join(LINES[item][2].split(",")))
                    elif "." in split:
                        DR_ERR = float(LINES[item][2])
                    else:
                        raise ReadingParserError(f"Ошибка получения переменной $DR_ERR")

            elif LINES[item][0] == '$GAIN:':
                if len(LINES[item]) >= 3:
                    gain = int(LINES[item][2])
            elif LINES[item][0] == '$HIGH_VOLTAGE:':
                if len(LINES[item]) >= 3:
                    HV = int(LINES[item][2])
            elif LINES[item][0] == '$LOWER_LIMIT:':
                if len(LINES[item]) >= 3:
                    Lower_Limit = int(LINES[item][2])
            elif LINES[item][0] == '$STAB_GAIN:':
                if len(LINES[item]) >= 3:
                    SGain = int(LINES[item][2])
            elif LINES[item][0] == '$TIME_DIV:':
                if len(LINES[item]) >= 3:
                    Time_div = int(LINES[item][2])
            elif LINES[item][0] == '$UPPER_LIMIT:':
                if len(LINES[item]) >= 3:
                    Upper_Limit = int(LINES[item][2])

        GPS['Longitude'] = GPS_LON
        GPS['Latitude'] = GPS_LAT
        GPS['Alternate'] = GPS_ALT
        GPS['Speed'] = GPS_SPD
        GPS['Direction'] = GPS_DIR
        GPS['Valid'] = GPS_VLD

        # ----------------------------------- Data (dict) ----------------------------------- #
        data["Counts"] = counts
        data['Date manufacture'] = DATE_MANUFACT
        data['GPS'] = GPS
        data['Energy'] = energy
        data['Sigma'] = resultion
        data['Time of collection'] = Time
        data['Realtime'] = realtime
        data['Activity'] = Activity
        data['Remark'] = remark
        data['Weight'] = weight
        data['Date Measurement'] = Date
        data['Dose rate'] = Dose
        data['Dose rate error'] = DR_ERR
        data['Gain'] = gain
        data['High Voltage'] = HV
        data['Lower Limit'] = Lower_Limit
        data['Upper Limit'] = Upper_Limit
        data['Stabilization Gain'] = SGain
        data['Time division'] = Time_div
        data['Temperature'] = temp
        data['Radionuclides'] = radionuclides
        data['cps'] = cps
        data['Activity result'] = ACTIVITYRESULT
        data['Effective activity result'] = EFFECTIVEACTIVITYRESULT
        data['Spectrum proceseed'] = SPECTRUMPROCESSED
        data['BGNDSUBTRACTED'] = BGNDSUBTRACTED
        data['ENCRYPTED'] = ENCRYPTED
        data['Type detector unit'] = DU
        data['Geometry'] = geometry
        data['' ] = firmware

    return data


if __name__ == '__main__':
    path = r'D:\ATOMTEXSPECTR\tests\spectrum\ats_sample.ats'
    print(reading(path))