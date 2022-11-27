from enum import Enum

class TextOps(Enum):
    """
    Description: List of available operations for text cleaning.

    Attributes:
        LOWER        : Lowers the entire corpus
        UNICODE      : Removes unwanted unicode characters, punctuations, emojis, URLS, etc.
        STOP_WORDS   : Removes common English words (nltk.corpus.stopwords)
        LEMMATIZE    : Lemmatizes words (nltk.stem.WordNetLemmatizer)
        STEM         : Extracts root stem of the words (nltk.stem.porter.PorterStemmer)
        DIGITS       : Removes the digits from the data.
        PUNCTUATIONS : Removes the punctuations from the data except (').
    """

    LOWER        = 0
    UNICODE      = 1
    ASCII        = 2
    STOP_WORDS   = 3
    LEMMATIZE    = 4
    STEM         = 5
    DIGITS       = 6
    PUNCTUATIONS = 7
