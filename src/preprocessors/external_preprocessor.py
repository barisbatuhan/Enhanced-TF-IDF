import re

from .base_preprocessor import BasePreprocessor

class ExternalPreprocessor(BasePreprocessor):

    def __init__(self, fn):
        """
        Description: Directly sets the fn that is an external callable
        """
        assert callable(fn)
        self.fn = fn