from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.text_processor import TextProcessor

class TF_IDF(TfidfVectorizer):
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
        txt_processor :TextProcessor,
        analyzer :str="word", 
        stop_words :list=None, 
        ngram_range :tuple=(1, 1),
        max_df :float=1.0,
        min_df :float=0.0,
        ):

        """
        Description: Constructor of the TF_IDF Object. 

        Inputs:
            txt_processor (TextProcessor) : Text processor module, that is initialized with the corpus.
            analyzer (string)             : Whether the feature should be made of word or character n-grams.
                                            Options: ["word", "char", "char_wb"]
            stop_words (List[string])     : Additional stop words (other than NLTK) to exclude that is 
                                            specific to the corpus.
            ngram_range (Tuple[int, int]) : The lower and upper boundary of the range of n-values for 
                                            different n-grams to be extracted.
            max_df (float)                : Ignore terms that have a document frequency strictly higher 
                                            than the given threshold. Acceptable range: (0.0, 1.0]
            min_df (float)                : Ignore terms that have a document frequency strictly lower 
                                            than the given threshold. Acceptable range: [0.0, 1.0)
        
        Note: Please note that this module will be used to initialized and run with the outputs of the
              custom-designed TextProcessor method. Thus, no additional preprocessing steps will be 
              executed in this module.
        """
        assert max_df > 0.0 and max_df <= 1.0
        assert min_df >= 0.0 and min_df < 1.0
        assert analyzer in ["word", "char", "char_wb"]

        self.txt_processor = txt_processor
        
        super().__init__(
            input="content",
            encoding='utf-8',
            lowercase=False,
            analyzer=analyzer,
            stop_words=stop_words,
            ngram_range=ngram_range,
            vocabulary=txt_processor.vocab,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False)

    def fit_transform(self):
        return super().fit_transform(self.txt_processor.get_docs())

