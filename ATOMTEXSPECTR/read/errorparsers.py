"""
Detection of errors and warnings for input/output (I/O).
The warnings module is useful when it is necessary to warn the user
about any condition in the program and this condition does not require creation
exceptions and program termination.
"""
import warnings
# warnings.filterwarnings("ignore")
warnings.simplefilter("always", DeprecationWarning)
class ReadingParserWarning(UserWarning):
    """
    Warnings may inevitably occur during file parsing.
    """
    pass
class ReadingParserError(Exception):
    """
    A failure that occurred during parsing.
    """
    pass