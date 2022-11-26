from nltk.stem.porter import PorterStemmer
from .base_tokenizer import BaseTokenizer

class StemTokenizer(BaseTokenizer):
    """
    Desription: Applies stemming operation of 'PorterStemmer' in NLTK as tokenization.
    """
    def __init__(self):
        self.tokenizer = PorterStemmer()
        self.tokenizer_fn = self.tokenizer.stem