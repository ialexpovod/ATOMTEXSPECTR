"""
This module is designed for parsing a file-spectrum with a .ats resolution.
"""
# Imported modules for parsing
import os
import numpy
from ATOMTEXSPECTR.encoding import UniDet
import dateutil.parser
from errorparsers import ReadingParserError

def reading(
        filename,
        deb=False
        ):
    """
    Parse the ASCII .ats file and return a dictionary of data
    :param filename:        Имя файла (строковый тип данных). The filename of the CNF file to read.
    :param deb:             bool
                            Следует ли выводить отладочную (debugging) информацию. По умолчанию False.
    :return:                Dictionary of data in spectrum
    """
    print('Filename file-spectrum:' + filename)  # Вывод названия файла
    namefile, extension = os.path.splitext(filename)
    # Check encoding file-spectrum
    if extension != '.ats':
        raise ReadingParserError('Формат файла неверный' + extension)
    # Check compliance with the format file-spectrum
    # This file-spectrum in conditional statement foe ".ats"
    enc = UniDet.encodingfile(filename)
    # Initialized dictionary for filling in the spectrum data during file parsing
    data = dict()
    sp_start_count = None
    time = None
    energy = list()
    resultion = list()
    counts = list()
    GPS = dict()
    with open(
            filename,
            encoding=enc
        ) as file_spectrum:
        # Список с компонентами из строк файла
        lines = [LINE.strip().split(" ") for LINE in file_spectrum.readlines()]
        for item in range(len(lines)):
            if lines[item][0] == "TIME":
                time = int(lines[item][2])
                if deb:
                    print(time)
            # elif lines[item][0] == 'ACTIVITY':
            #     activity = float(lines[item][2])
            # elif lines[item][0] == 'REMARK':
            #     if len(lines[item]) >= 3:
            #         remark = lines[item][2]
            #     else:
            #         remark = None
            # elif lines[item][0] == 'GEOMETRY':
            #     if len(lines[item]) >= 3:
            #         geometry = lines[item][2]
            #     else:
            #         geometry = None
            # elif lines[item][0] == 'WEIGHT':
            #     if len(lines[item]) >= 3:
            #         weight = float(lines[item][2])
            elif lines[item][0] == 'DATE':
                sp_start_count = dateutil.parser.parse(" ".join(lines[item][2:]))
            # elif lines[item][0] == 'RADIATION':
            #     if len(lines[item]) >= 3:
            #         radiation = lines[item][2]

            # GPS
            elif lines[item][0] == 'GPS_LAT':
                if len(lines[item]) >= 3:
                    latitude = float(lines[item][2])
            elif lines[item][0] == 'GPS_LON':
                if len(lines[item]) >= 3:
                    longitude = float(lines[item][2])
            elif lines[item][0] == 'GPS_ALT':
                if len(lines[item]) >= 3:
                    alternate = float(lines[item][2])
            elif lines[item][0] == 'GPS_SPD':
                if len(lines[item]) >= 3:
                    speed = float(lines[item][2])
            elif lines[item][0] == 'GPS_DIR':
                if len(lines[item]) >= 3:
                    direction = float(lines[item][2])
            elif lines[item][0] == 'GPS_VLD':
                if len(lines[item]) >= 3:
                    valid = float(lines[item][2])
            # elif lines[item][0] == 'DOSERATE':
            #     if len(lines[item]) >= 3:
            #         dr = float(lines[item][2])
            # elif lines[item][0] == 'DU_TYPE':
            #     if len(lines[item]) >= 3:
            #         du_name = lines[item][2]
            # elif lines[item][0] == 'SCALE_MODE':
            #     if len(lines[item]) >= 3:
            #         scale = float(lines[item][2])
            # elif lines[item][0] == 'SERIAL':
            #     if len(lines[item]) >= 3:
            #         serial = int(lines[item][2])
            # elif lines[item][0] == 'FIRMWARE':
            #     if len(lines[item]) >= 3:
            #         firmware = float(lines[item][2])
            # elif lines[item][0] == 'HARDWARE':
            #     if len(lines[item]) >= 3:
            #         harware = float(lines[item][2])
            # elif lines[item][0] == 'DISTANCE':
            #     if len(lines[item]) >= 3:
            #         distance = float(lines[item][2])
            # elif lines[item][0] == 'SWIDTH':
            #     if len(lines[item]) >= 3:
            #         swidth = float(lines[item][2])
            # elif lines[item][0] == 'CPSOUTOFRANGE':
            #     if len(lines[item]) >= 3:
            #         cpsoutfrange = float(lines[item][2])
            # elif lines[item][0] == 'DATE_MANUFACT':
            #     if len(lines[item]) >= 3 and len(lines[item][2]) == 7:
            #         date_manufact = dateutil.parser.parse(lines[item][2]).strftime("%Y-%m")
            #     else:
            #         raise ReadingParserError(f"Unknown data value: {lines[item][2]}")
            # realtime
            elif lines[item][0] == 'REALTIME':
                if len(lines[item]) >= 3:
                    realtime = int(lines[item][2])
            elif lines[item][0] == 'CPS':
                if len(lines[item]) >= 3:
                    cps = float(lines[item][2])
            # elif lines[item][0] == 'ACTIVITYRESULT':
            #     if len(lines[item]) >= 3:
            #         activity_result = float(lines[item][2])
            #     else:
            #         activity_result = None
            # elif lines[item][0] == 'EFFECTIVEACTIVITYRESULT':
            #     if len(lines[item]) >= 3:
            #         ef_activity_result = lines[item][2]
            #     else:
            #         ef_activity_result = None
            # elif lines[item][0] == 'MIX':
            #     if len(lines[item]) >= 3:
            #         mix = lines[item][2]
            #     else:
            #         mix = None
            # elif lines[item][0] == 'SPECTRUMPROCESSED':
            #     if len(lines[item]) >= 3:
            #         spectrum_processed = float(lines[item][2])
            # elif lines[item][0] == 'BGNDSUBTRACTED':
            #     if len(lines[item]) >= 3:
            #         bgno_substracted = float(lines[item][2])
            # elif lines[item][0] == 'ENCRYPTED':
            #     if len(lines[item]) >= 3:
            #         encrypted = float(lines[item][2])
            # elif lines[item][0] == 'NEUTRON_CPS':
            #     if len(lines[item]) >= 3:
            #         neutron_cps = float(lines[item][2])
            # elif lines[item][0] == 'NEUTRON_COUNT':
            #     if len(lines[item]) >= 3:
            #         neutron_count = int(lines[item][2])
            # elif lines[item][0] == 'NEUTRON_DOSERATE':
            #     if len(lines[item]) >= 3:
            #         neutron_doserate = float(lines[item][2])
            # elif lines[item][0] == 'STATUS_OF_HEALTH':
            #     if len(lines[item]) >= 3:
            #         status_of_health = lines[item][2]
            # GPS
            # elif lines[item][0] == 'GPS_RMCHOUR':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCHOUR = float(lines[item][2])
            # elif lines[item][0] == 'GPS_RMCMINUTE':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCMINUTE = float(lines[item][2])
            # elif lines[item][0] == 'GPS_RMCMINUTE':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCMINUTE = float(lines[item][2])
            # elif lines[item][0] == 'GPS_RMCSECOND':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCSECOND = float(lines[item][2])
            # elif lines[item][0] == 'GPS_RMCDAY':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCDAY = float(lines[item][2])
            # elif lines[item][0] == 'GPS_RMCMONTH':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCMONTH = float(lines[item][2])
            # elif lines[item][0] == 'GPS_RMCYEAR':
            #     if len(lines[item]) >= 3:
            #         GPS_RMCYEAR = float(lines[item][2])
            # elif lines[item][0] == 'GPS_GSAPDOP':
            #     if len(lines[item]) >= 3:
            #         GPS_GSAPDOP = float(lines[item][2])
            # elif lines[item][0] == 'GPS_GSAHDOP':
            #     if len(lines[item]) >= 3:
            #         GPS_GSAHDOP = float(lines[item][2])
            # elif lines[item][0] == 'GPS_GSAVDOP':
            #     if len(lines[item]) >= 3:
            #         GPS_GSAVDOP = float(lines[item][2])
            elif lines[item][0] == 'RADIONUCLIDES':
                radionuclides = lines[item][2:]
            elif lines[item][0] == 'TEMPERATURE':
                temp = float(lines[item][2])
            elif lines[item][0] == 'ECALIBRATION':
                ecalibration = int(lines[item][2])
                if ecalibration != 0 or ecalibration is not None:
                    j = 0
                    while j <= ecalibration - 1:
                        item += 1
                        # print(lines[item][0])
                        energy = numpy.append(energy, float(lines[item][0]))
                        j += 1
                else:
                    raise ReadingParserError(f'Error in obtaining energy calibration data: {ecalibration}')

            elif lines[item][0] == 'RCALIBRATION':
                res = int(lines[item][2])
                if res != 0 or res is not None:
                    j = 0
                    while j <= res - 1:
                        item += 1
                        resultion = numpy.append(resultion, float(lines[item][0]))
                        j += 1
                else:
                    raise ReadingParserError(f'Error in obtaining calibration data by resolution: {res}')

            elif lines[item][0] == 'SPECTR':
                spectr = int(lines[item][2])
                if spectr != 0 or spectr is not None:
                    j = 0
                    while j <= spectr - 1:
                        item += 1
                        counts = numpy.append(counts, float(lines[item][0]))
                        j += 1
                else:
                    raise ReadingParserError(f'Error receiving data on counts: {spectr}')

            # elif lines[item][0] == '$DR_ERR:':
            #     if len(lines[item]) >= 3:
            #         split = lines[item][2]
            #         if "," in split:
            #             dr_err = float(".".join(lines[item][2].split(",")))
            #         elif "." in split:
            #             dr_err = float(lines[item][2])
            #         else:
            #             raise ReadingParserError(f"Error getting a variable$DR_ERR")

            # elif lines[item][0] == '$GAIN:':
            #     if len(lines[item]) >= 3:
            #         gain = int(lines[item][2])
            # elif lines[item][0] == '$HIGH_VOLTAGE:':
            #     if len(lines[item]) >= 3:
            #         hv = int(lines[item][2])
            # elif lines[item][0] == '$LOWER_LIMIT:':
            #     if len(lines[item]) >= 3:
            #         lower_Limit = int(lines[item][2])
            # elif lines[item][0] == '$STAB_GAIN:':
            #     if len(lines[item]) >= 3:
            #         s_gain = int(lines[item][2])
            # elif lines[item][0] == '$TIME_DIV:':
            #     if len(lines[item]) >= 3:
            #         time_div = int(lines[item][2])
            # elif lines[item][0] == '$UPPER_LIMIT:':
            #     if len(lines[item]) >= 3:
            #         upper_Limit = int(lines[item][2])

        GPS['Longitude'] = longitude
        GPS['Latitude'] = latitude
        GPS['Alternate'] = alternate
        GPS['Speed'] = speed
        GPS['Direction'] = direction
        GPS['Valid'] = valid

        # ----------------------------------- Data (dict) ----------------------------------- #
        data["counts"] = counts
        data['sp_start_count'] = sp_start_count
        data['Temperature'] = temp
        data['energy'] = energy
        data['sigma'] = resultion
        data['measuretime'] = time
        data['actualtime'] = realtime
        data['cps'] = cps
        # data['Date manufacture'] = date_manufact
        data['GPS'] = GPS
        # data['Activity'] = activity
        # data['Remark'] = remark
        # data['Weight'] = weight
        # data['Dose rate'] = dr
        # data['Gain'] = gain
        # data['High Voltage'] = HV
        # data['Lower Limit'] = Lower_Limit
        # data['Upper Limit'] = Upper_Limit
        # data['Stabilization Gain'] = SGain
        # data['Time division'] = Time_div
        data['Radionuclides'] = radionuclides
        # data['Activity result'] = activity_result
        # data['Effective activity result'] = ef_activity_result
        # data['Spectrum proceseed'] = spectrum_processed
        # data['BGNDSUBTRACTED'] = bgno_substracted
        # data['ENCRYPTED'] = encrypted
        # data['Type detector unit'] = du_name
        # data['Geometry'] = geometry
        # data['firmware'] = firmware
    return data


if __name__ == '__main__':
    path = r'D:\ATOMTEXSPECTR\tests\spectrum\ats_sample.ats'
    print(
            reading(path)
          )