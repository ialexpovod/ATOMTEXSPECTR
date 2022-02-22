"""
This module is designed for parsing a file-spectrum with a .spe resolution.
"""
# Imported modules for parsing
import os
import numpy
from ATOMTEXSPECTR.encoding import UniDet
import dateutil.parser
from ATOMTEXSPECTR.read import errorparsers

def reading(
            filename,
            deb=False
            ):
    """
    Parse the ASCII .spe file-spectrum and return a dictionary of data.

    :param filename:Имя файла (строковый тип данных). The filename of the CNF file to read.
    :param deb: bool
    Следует ли выводить отладочную (debugging) информацию. По умолчанию False.
    :return: Dictionary of data in spectrum
    """

    print('Filename file-spectrum:' + filename)  # Return in console filename input file-spectrum
    namefile, extension = os.path.splitext(filename)
    # Check encoding file-spectrum
    enc_file = UniDet.encodingfile(filename)
    # Check compliance with the format file-spectrum
    # This file-spectrum in conditional statement foe ".spe"
    if extension != '.spe':
        raise errorparsers.ReadingParserError("The format file-spectrumincorrect" + extension)
    # Initialized dictionary for filling in the spectrum data during file parsing
    data = dict()
    sigma = list()
    sp_start_count = None
    counts = list()
    channels = list()
    mca_166_id = list()
    energy = list()
    radionuclides = list()
    GPS = dict()

    with open(
            filename,
            encoding=enc_file
            ) as file_spectrum:
        # List with component from file-spectrum lines
        lines = [LINE.strip() for LINE in file_spectrum.readlines()]
        for item in range(len(lines)):
            if lines[item] == '$MCA_166_ID:':
                item = item + 1
                mca_166_id.append(item)
            elif lines[item] == "$DATE_MEA:":
                item += 1
                sp_start_count = dateutil.parser.parse(lines[item])
                if deb:
                    print(sp_start_count)
            elif lines[item] == "$DATA:":
                item += 1
                firstchannel = float(lines[item].split(" ")[0])
                if firstchannel != 0:
                    raise errorparsers.ReadingParserError(f"First channel is not 0: {firstchannel}")
                numberchannel = float(lines[item].split(" ")[1])
                if deb:
                    print(firstchannel, numberchannel)
                j = firstchannel
                while j <= firstchannel + numberchannel:
                    item += 1
                    counts = numpy.append(counts, int(lines[item]))
                    channels = numpy.append(channels, j)
                    j += 1

            # elif lines[item] == '$ENER_FIT:':
            #     item += 1
            #     ene_fit = lines[item].split(" ")
            #     energyfit = [float(index) for index in ene_fit]

            # elif lines[item] == '$ROI:':
            #     item += 1
            #     roi = lines[item]

            elif lines[item] == '$ENER_TABLE:':
                item += 1
                len_channel = int(lines[item].split(" ")[0])
                first_channel = float(lines[item + 1].split(" ")[0])
                j = first_channel
                while j < len_channel:
                    item += 1
                    energy = numpy.append(energy, float(lines[item].split(" ")[1]))
                    j += 1

            elif lines[item] == '$SIGM_DATA:':
                item += 1
                len_channel = int(lines[item].split(" ")[0])
                first_channel = float(lines[item + 1].split(" ")[0])
                j = first_channel
                while j < len_channel:
                    item += 1
                    sigma = numpy.append(sigma, float(lines[item].split(" ")[1]))
                    j += 1
            elif lines[item] == '$TEMPERATURE:':
                temp = float(lines[item + 1])
            # elif lines[item] == '$SCALE_MODE:':
            #     scale = float(item + 1)
            elif lines[item] == '$DOSE_RATE:':
                dr = float(lines[item + 1])
                if dr is None or 0:
                    errorparsers.ReadingParserWarning('Error in obtaining the dose rate value:' + str(dr))
            # elif lines[item] == '$DU_NAME:':
            #     du_name = lines[item + 1]
            elif lines[item] == '$RADIONUCLIDES:':
                radionuclides.append(lines[item + 1].split(" "))
            elif lines[item] == '$ACTIVITYRESULT:':
                item += 1
                activity = lines[item]
                if deb and len(activity) == 0:
                    print('Error getting information about activity, no data.')
            elif lines[item] == '$EFFECTIVEACTIVITYRESULT:':
                ef_activity_result = lines[item + 1]
                if len(ef_activity_result) == 0 and deb:
                    print('Error in obtaining information about the effectiveness of registration, there is no data.')
            elif lines[item] == '$GEOMETRY:':
                geometry = lines[item + 1]
                if len(geometry) == 0 and deb:
                    print('Error getting information about geometry, no data.')
            elif lines[item] == '$GAIN:':
                gain = lines[item + 1]
            # GPS data
            elif lines[item] == '$GPS:':
                longitude = lines[item + 1].split(" ")[1]
                latitude = lines[item + 2].split(" ")[1]
                alternate = lines[item + 3].split(" ")[1]
                speed = lines[item + 4].split(" ")[1]
                direction = lines[item + 5].split(" ")[1]
                valid = lines[item + 6].split(" ")[1]
                GPS['Longitude'] = longitude
                GPS['Latitude'] = latitude
                GPS['Alternate'] = alternate
                GPS['Speed'] = speed
                GPS['Direction'] = direction
                GPS['Valid'] = valid
            elif lines[item] == "$MEAS_TIM:":
                item += 1
                meatime = int(lines[item].split(" ")[0])
                sectime = int(lines[item].split(" ")[1])
            elif lines[item].startswith("$"):
                key = lines[item][1:].rstrip(":")
                item += 1
                values = []
                while item < len(lines) and not lines[item].startswith("$"):
                    values.append(lines[item])
                    item += 1
                if item < len(lines):
                    if lines[item].startswith("$"):
                        item -= 1
                if len(values) == 1:
                    values = values[0]
                data[key] = values
            item += 1
    data["counts"] = counts
    data["sp_start_count"] = sp_start_count
    data['measuretime'] = meatime
    data['actualtime'] = sectime
    # data['GPS'] = GPS
    data['temp'] = temp
    data['gain'] = gain
    data['energy'] = energy
    data['sigma'] = sigma
    return data


if __name__ == '__main__':
    path = r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe'
    file = reading(path)
    print(file)
