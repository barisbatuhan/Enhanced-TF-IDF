import re

from .base_preprocessor import BasePreprocessor

class MultiPreprocessor(BasePreprocessor):
    """
    Description: combines multiple preprocessors and applies them
                 one-by-one
    """

    def __init__(self, preprocessor_list):
        self.preprocessors = preprocessor_list
        self.fn = self.process_multi

    def process_multi(self, text :str):
        for p in self.preprocessors:
            text = p(text)
        return text