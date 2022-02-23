"""
Detection of errors and warnings for input/output (I/O).
The warnings module is useful when it is necessary to warn the user
about any condition in the program and this condition does not require
creation exceptions and program termination.
"""

class spectrum_error(Exception):
    """
    Exception raised by spectrum.py
    """
    pass

class spectrum_warning(UserWarning):
    """
    Warnings displayed by spectrum.py
    """
    pass

class uncalibrated_error(spectrum_error):
    """
    Raised when an uncalibrated spectrum is treated as calibrated
    """
    pass



class plot_spectrum:
    """
    Class for parsing spectrum.
    """


class inaccuracy_error(Exception):
    """
    Raised when  inaccuracy (in this case, uncertainties)
    are badly specified in an input
    """
    pass

class plot_error(Exception):
    '''
    Exception raised by plot.py
    '''
    pass
