import re

from .base_preprocessor import BasePreprocessor

class DigitPreprocessor(BasePreprocessor):
    """
    Description: Given a text, removes all the digits in the text.
    """

    def __init__(self):
        self.fn = self.process_digits

    def process_digits(self, text :str):
        return re.sub(r'[ ]+', ' ', re.sub(r'[0-9]+', '', text)).strip()