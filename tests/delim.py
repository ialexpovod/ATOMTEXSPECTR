delim = '|'
l = ["1","2", "3"]
# print(delim.join(l))
# print("{:>6}".format("") + "|", "{:<6}".format("") + "|")
header = ["{:>{}}".format(" ", len(l)), "{:>6}".format(" ", len(l))]
print(delim.join(header))
import platform
print(platform.system())
"""
Spectrum:
    Name spectrum::                          241am.spe
    Path spectrum::                          D:\PycharmProject\ATOMTEXSPECTR\tests\spectrum\
    Calibrated:                              True
    Start time:                              2021-12-27 16:00:46
    Stop time:                               2021-12-27 16:04:10
    Measurement:                             200.0
    Numbers of channels:                     1024
    Sum counts:                              (1.881+/-0.004)e+05
    Spectrum cps:                            940.4+/-2.2
"""
