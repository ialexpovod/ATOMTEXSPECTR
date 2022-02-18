import numpy
from ATOMTEXSPECTR import spectr as sp
from uncertainties import ufloat, unumpy
import matplotlib.pyplot
from ATOMTEXSPECTR.read import spe
spectr_1 = sp.Spectr.import_file(r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe')

y = numpy.random.normal(10, 1, 10)
# arr = unumpy.uarray([1, 2], [0.01, 0.002])
#
# print(unumpy.nominal_values(arr))
# print(unumpy.std_devs(arr))

# x = numpy.arange(0, 100)
# matplotlib.pyplot.plot(x, y)
# matplotlib.pyplot.show()
# print(y)
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

    matplotlib.pyplot.show()
    # e = r'D:\ATOMTEXSPECTR\tests\spectrum\BDKG11M_sample.spe'
    # encoding = UniDet.encodingfile(path)
    # file = spe.reading(e)
    # # print(file.keys())
    # # print(file['energy'])
    # print(file)
