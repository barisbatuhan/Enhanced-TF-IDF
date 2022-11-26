# STD Libraries
from typing import List, Tuple, Set, Dict, Union, Callable
# Custom Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
# User-defined Files
from ..tokenizers import LemmaTokenizer, StemTokenizer
from ..preprocessors import DigitPreprocessor, PuncPreprocessor, MultiPreprocessor, ExternalPreprocessor
from ..utils import TextOps, ENGLISH_STOP_WORDS

class TfIdfModel(TfidfVectorizer):
    """
        Description: TF_IDF Model Object Class. The object is inherited from 
                     scikit-learn's TfidfVectorizer. The methods except the
                     constructor are the same with the original parent class,
                     which can be accessed from this link:
                     https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        Attributes:
            vocabulary (dict)                   : A mapping of terms to feature indices.
            fixed_vocabulary_ (bool)            : True if a fixed vocabulary of term to indices 
                                                  mapping is provided by the user.
            idf_ (array of shape (n_features,)) : Inverse document frequency vector.
            stop_words (Set[string])            : additional stop words that are given in constructor 
                                                  or occurred in too many/few documents (max_df & min_df).                         
    """
    
    def __init__(
        self, 
        op_set       : Set[TextOps],
        analyzer     : Union[str, Callable]       = "word",
        stop_words   : Union[str, List[str]]      = None,
        ngram_range  : Tuple[int, int]            = (1, 1),
        max_df       : Union[int, float]          = 1.0,
        min_df       : Union[int, float]          = 0.0,
        max_features : int                        = None,
        vocabulary   : Union[List[str], Set[str]] = None,
        binary       : bool                       = False,
        **kwargs,
        ):

        """
        Description: Constructor of the TF_IDF Object. 

        Inputs:
            op_set (Set[TextOps])         : Set of text processing operations to be applied to data.
            analyzer (string)             : Whether the feature should be made of word or character n-grams.
                                            Options: ["word", "char", "char_wb"]
            stop_words (List[string])     : Stop words to exclude that is specific to the corpus. If set
                                            to 'default', default stop words will be used.
            ngram_range (Tuple[int, int]) : The lower and upper boundary of the range of n-values for 
                                            different n-grams to be extracted.
            max_df (Union[int, float])    : Ignore terms that have a document frequency (for float) / count 
                                            (for int) strictly higher than the given threshold. 
                                            Acceptable range: (0.0, 1.0] for float, > 0 for int.
            min_df (Union[int, float])    : Ignore terms that have a document frequency (for float) / count 
                                            (for int) strictly lower than the given threshold. 
                                            Acceptable range: [0.0, 1.0) for float, >= 0 for int.
            max_features (int)            : If not None, build a vocabulary that only consider the top 
                                            max_features ordered by term frequency across the corpus. This 
                                            parameter is ignored if vocabulary is not None.
            vocabulary (Union[List[str], Set[str]]) : An iterable over terms. If not given, a vocabulary is 
                                            determined from the input documents.
            binary (bool)                 : If True, all non-zero term counts are set to 1. This does not 
                                            mean outputs will have only 0/1 values, only that the tf term 
                                            in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs).
        """

        assert max_features is None or max_features > 0, "'max_features' should be a positive integer !"
        
        assert callable(analyzer) or analyzer in ["word", "char", "char_wb"], \
            "Analyzer can be a callable or one of [word, char, char_wb] !"
        
        assert type(max_df) != float or (max_df > 0.0 and max_df <= 1.0), \
            "If float, max_df should be in range (0.0, 1.0] !"
        assert type(max_df) != int or max_df > 0, \
            "If int, max_df should be greater than 0 !"
        assert type(min_df) != float or (min_df >= 0.0 and min_df < 1.0), \
            "If float, min_df should be in range [0.0, 1.0) !"
        assert type(min_df) != int or min_df >= 0, \
            "If int, min_df should be greater than equal to 0 !"
        
        self.op_set = op_set
        
        super().__init__(
            input="content",
            strip_accents=self._get_strip_accent(),
            lowercase=TextOps.LOWER in op_set,
            tokenizer=self._get_tokenizer(),
            analyzer=analyzer,
            stop_words=self._get_stop_words(stop_words),
            ngram_range=ngram_range,
            vocabulary=vocabulary,
            max_df=max_df,
            min_df=min_df,
            binary=binary,
            **kwargs
            )

    
    def train(self, corpus :List[str]):
        """
        Description: An alias to the 'fit_transform' method in scikit-learn.

        Inputs:
            corpus (List[string]) : list of string documents.

        Outputs:
            X (np.ndarray)        : 2D numpy Tf-idf-weighted document-term matrix
        """
        return super().fit_transform(corpus).toarray()


    def infer(self, corpus :List[str]):
        """
        Description: An alias to the 'transform' method in scikit-learn.

        Inputs:
            corpus (List[string]) : list of string documents.

        Outputs:
            X (np.ndarray)        : 2D numpy Tf-idf-weighted document-term matrix
        """
        return super().transform(corpus).toarray()

    def get_feature_names(self):
        """
        Description: An alias to the 'get_feature_names_out' method in scikit-learn.

        Inputs:
            corpus (List[string]) : list of string documents.

        Outputs:
            X (List[Any])         : Feature words selected from the process
        """
        return super().get_feature_names_out().tolist()

    
    """ --------------------------------------------------------------------------------------
    ----- GETTERS OF THE SKLEARN'S TFIDFVECTORIZER
    -------------------------------------------------------------------------------------- """
    
    def _get_tokenizer(self):
        assert TextOps.LEMMATIZE not in self.op_set or TextOps.STEM not in self.op_set, \
            "Both Lemmatization and Stemmer cannot be applied together !"
        
        if TextOps.LEMMATIZE in self.op_set:
            return LemmaTokenizer()
        elif TextOps.STEM in self.op_set:
            return StemTokenizer()
        else:
            return None
    

    def _get_strip_accent(self):
        assert TextOps.ASCII not in self.op_set or TextOps.UNICODE not in self.op_set, \
            "Both ASCII and UNICODE cannot be applied together !"
        
        if TextOps.ASCII in self.op_set:
            return "ascii"
        elif TextOps.UNICODE in self.op_set:
            return "unicode"
        else:
            return None

    def _get_stop_words(self, stop_words):
        """
        Description: Handles three different approaches:
            1) If TextOps.STOP_WORD is not in op_set, then no stop words will be applied
            2) If stop_words is a string and its value is 'default', detault stop words will be applied.
            3) If a list of stop words are given, then these will be applied.

        Outputs:
            stop_words (List[string]) : final stop words to be applied to the corpus
        """
        assert type(stop_words) != str or stop_words == "default", \
            "If stop_words is given as a string, only supported value is 'default' !"
        assert TextOps.STOP_WORDS not in self.op_set or stop_words is not None, \
            "If stop words will be discarded from the data, stop_words must be specified !"

        if TextOps.STOP_WORDS not in self.op_set:
            return None
        elif type(stop_words) == str and stop_words == "default":
            return ENGLISH_STOP_WORDS
        else:
            return stop_words
    
    def build_preprocessor(self):
        """
        Overrides of its parent's method since additional preprocessing operations are also required.
        """
        preprocessors = [ExternalPreprocessor(super().build_preprocessor())]
        if TextOps.DIGITS in self.op_set:
            preprocessors.append(DigitPreprocessor())
        if TextOps.PUNCTUATIONS in self.op_set:
            preprocessors.append(PuncPreprocessor())
        return MultiPreprocessor(preprocessors)
