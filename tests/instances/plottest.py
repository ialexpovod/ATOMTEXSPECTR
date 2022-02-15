import numpy
from ATOMTEXSPECTR import spectr as sp
import matplotlib.pyplot
from ATOMTEXSPECTR.read import spe
spectr_1 = sp.Spectr.import_file(r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe')




if __name__ == '__main__':
    spectr_1.plot()
    matplotlib.pyplot.show()
    # e = r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe'
    # encoding = UniDet.encodingfile(path)
    # file = spe.reading(e)
    # # print(file.keys())
    # # print(file['energy'])
    # print(file)
