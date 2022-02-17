import numpy
from ATOMTEXSPECTR import spectr as sp
import matplotlib.pyplot
from ATOMTEXSPECTR.read import spe
spectr_1 = sp.Spectr.import_file(r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe')




if __name__ == '__main__':
    # spectr_1.set_xlim((0,200))
    print(spectr_1)
    # print(spectr_1.attrs)
    spectr_1.plot('-r', ymode = 'cnt',
                          yscale = 'linear',
                          title = 'Spectr_1',
                          linewidth = 1,
                          alpha = 0.5,
                          #edgecolor='black',
                          xlim = (125,170),
                          ylim = (0,200))

    # matplotlib.pyplot.show()
    # e = r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe'
    # encoding = UniDet.encodingfile(path)
    # file = spe.reading(e)
    # # print(file.keys())
    # # print(file['energy'])
    # print(file)
