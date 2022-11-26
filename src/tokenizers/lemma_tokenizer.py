import nltk
nltk.download('wordnet', quiet=True)

from nltk.stem import WordNetLemmatizer 

from .base_tokenizer import BaseTokenizer

class LemmaTokenizer(BaseTokenizer):
    """
    Desription: Applies lemmatizing operation of 'WordNetLemmatizer' in NLTK as tokenization.
    """
    def __init__(self):
        self.tokenizer = WordNetLemmatizer()
        self.tokenizer_fn = self.tokenizer.lemmatize