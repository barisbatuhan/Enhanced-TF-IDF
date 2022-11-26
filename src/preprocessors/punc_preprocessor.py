import re
import string

from .base_preprocessor import BasePreprocessor

class PuncPreprocessor(BasePreprocessor):
    """
    Description: given a text, removes all punctuations and additional whitespaces
                 created by this removal process
    """

    def __init__(self):
        self.fn = self.process_puncs

    def process_puncs(self, text :str):
        text = re.sub(r'[!"#$%&()*+,-.\'\/\\:;<=>?@[\]^_`{|}~]+', ' ', text)
        return re.sub(r'[ ]+', ' ', text).strip()

    