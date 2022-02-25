import numpy
from ATOMTEXSPECTR import spectrum as sp
import matplotlib.pyplot
from ATOMTEXSPECTR.read import spe
print("Reading '.spe' format:")
spectrum_Co_60 = sp.spectrum.import_file(r'D:\PycharmProject\ATOMTEXSPECTR\tests\spectrum\60co.spe')
spectrum_Cs_137 = sp.spectrum.import_file(r'D:\PycharmProject\ATOMTEXSPECTR\tests\spectrum\137cs.spe')
spectrum_Cd_109 = sp.spectrum.import_file(r'D:\PycharmProject\ATOMTEXSPECTR\tests\spectrum\109cd.spe')
spectrum_Th_228 = sp.spectrum.import_file(r'D:\PycharmProject\ATOMTEXSPECTR\tests\spectrum\228th.spe')
spectrum_Am_241 = sp.spectrum.import_file(r'D:\PycharmProject\ATOMTEXSPECTR\tests\spectrum\241am.spe')


print(spectrum_Am_241)


if __name__ == '__main__':
    None