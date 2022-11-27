import re

from .base_preprocessor import BasePreprocessor

class MultiPreprocessor(BasePreprocessor):
    """
    Description: Combines multiple preprocessors and applies them one-by-one.
    """

    def __init__(self, preprocessor_list: list):
        
        assert type(preprocessor_list) == list, "Preprocessor list is not a list !"
        assert preprocessor_list is not None, "List of preprocessors cannot be None !"
        assert len(preprocessor_list) > 0, "List of preprocessors cannot be empty !"
        
        for p in preprocessor_list:
            assert callable(p), "Parameter passed as preprocessor is not a callable !"

        self.preprocessors = preprocessor_list
        self.fn = self.process_multi

    def process_multi(self, text :str):
        for p in self.preprocessors:
            text = p(text)
        return text