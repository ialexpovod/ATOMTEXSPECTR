import warnings
"""
Определение ошибок и предупреждений для ввода-вывода (I/O).
Модуль warnings полезен, когда необходимо предупредить пользователя 
о каком-либо условии в программе и это условие не требует создания 
исключения и завершения программы.
"""
# warnings.filterwarnings("ignore")
warnings.simplefilter("always", DeprecationWarning)
class ReadingParserWarning(UserWarning):
    '''
    Предупреждения, возникающие во время парсинга.
    '''
    pass
class ReadingParserError(Exception):
    '''
    Сбой, возникший во время парсинга.
    '''
    pass

