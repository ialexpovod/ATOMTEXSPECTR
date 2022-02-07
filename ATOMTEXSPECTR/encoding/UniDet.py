def encodingfile(Path, NameFile):
  from chardet.universaldetector import UniversalDetector
  '''
    Функция возвращает кодировку указанного файла в формате строки (строковый тип даннных).
    :param Path: Директроия, где располагается файл.
    :param NameFile: Имя файла с расширением (namefile.txt)
    :return: Кодировка файла ('utf-8')
    '''
  enc = UniversalDetector()
  with open(Path + f'\{NameFile}', 'rb') as flop:
    for line in flop:
        enc.feed(line)
        if enc.done:
            break
    enc.close()
    return enc.result
if __name__ == '__main__':
    path = r''
    namefile = ''
    print(encodingfile(path, namefile)['encoding'])




