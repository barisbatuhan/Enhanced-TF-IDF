from typing import List

import nltk
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk import word_tokenize  

class BaseTokenizer:

    def __init__(self):
        self.tokenizer = None
        self.tokenizer_fn = None
    
    def __call__(self, text : str):
        """
        Description: Applies the tokenizer_fn that is set by other child constructors.
        """
        assert text is not None and len(text) > 0, "Text to tokenize cannot be None or empty !"
        assert self.tokenizer is not None, "Tokenizer cannot be empty !"
        assert callable(self.tokenizer_fn), "Tokenizer function must be a callable !"
        return [self.tokenizer_fn(t) for t in word_tokenize(text)]

    