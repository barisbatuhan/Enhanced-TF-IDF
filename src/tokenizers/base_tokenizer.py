from typing import List

import nltk
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk import word_tokenize  

class BaseTokenizer:

    def __init__(self):
        self.tokenizer = None
        self.tokenizer_fn = None
    
    def __call__(self, text : List[str]):
        """
        Description: Applies the tokenizer_fn that is set by other child constructors
        """
        assert self.tokenizer is not None
        assert self.tokenizer_fn is not None
        return [self.tokenizer_fn(t) for t in word_tokenize(text)]

    