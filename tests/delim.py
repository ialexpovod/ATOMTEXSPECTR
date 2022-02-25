delim = '|'
l = ["1","2", "3"]
# print(delim.join(l))
# print("{:>6}".format("") + "|", "{:<6}".format("") + "|")
header = ["{:>{}}".format(" ", len(l)), "{:>6}".format(" ", len(l))]
print(delim.join(header))
import platform
print(platform.system())