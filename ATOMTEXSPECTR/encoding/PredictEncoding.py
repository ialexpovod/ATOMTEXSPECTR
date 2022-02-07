def PredictEncoding(Path, Namefile, listraw):
    import chardet
    if listraw == None:
        listraw = 10
    with open(Path + f'\{Namefile}', 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join([f.readline() for _ in range(listraw)])
    return chardet.detect(rawdata), rawdata

path = r''
namefile = ''
print(PredictEncoding(path, namefile, listraw = 10))