import re
import copy
import string
from enum import Enum

from tqdm import tqdm

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


class TextOps(Enum):
    """
    Description: List of available operations for text cleaning

    Attributes:
        LOWER      : Lowers the entire corpus
        UNICODE    : Removes unwanted unicode characters, punctuations, emojis, URLS, etc.
        STOP_WORDS : Removes common English words (nltk.corpus.stopwords)
        LEMMATIZE  : Lemmatizes words (nltk.stem.WordNetLemmatizer)
        STEM       : Extracts root stem of the words (nltk.stem.porter.PorterStemmer)
        DIGITS     : Removes the digits from the data.
    """

    LOWER      = 0
    UNICODE    = 1
    STOP_WORDS = 2
    LEMMATIZE  = 3
    STEM       = 4
    DIGITS     = 5


class TextProcessor:
    """
    Description: Reads and processes the text data inside of a corpus.

    Attributes:
        filepath (string)         : Keeps the corpus '.txt' file path.
        op_set (Set[TextOps])     : Sequential list of operations to perform on text.
        docs (List[string])       : Keeps the documents in the corpus as a list.
        vocab (Dict[string, int]) : Keeps vocabulary in the document along with their ids.
        min_occur_cnt (int)       : Words that have number of calls in the corpus lower than
                                    'min_occur_cnt' are replaced with '<unk>' token.
    """

    def  __init__(self, corpus_file :str, op_set :set, vocab :set=None, min_occur_cnt :int=-1, vocab_size :int=-1):
        """
        Description: Constructor method of 'TextProcessor' object that reads a corpus and initializes vocabulary.

        Inputs:
            corpus_file (string)      : The '.txt' file path of the data corpus. 
                                        Each document is placed to a new line.
            op_set (Set[TextOps])     : Set of text processing operations to perform at each document.
            vocab (Set[string])       : Keeps vocabulary in the document. If given, no new vocabulary is 
                                        constructed, else the vocabulary is created from scratch.
            min_occur_cnt (int)       : Words that have number of calls in the corpus lower than
                                        'min_occur_cnt' are replaced with '<unk>' token. If -1,
                                        then no thresholding is applied. Both 'min_occur_cnt' and
                                        'vocab_size' cannot be set together.
            vocab_size (int)          : Number of words that are accepted to the vocabulary. If -1,
                                        then no thresholding is applied. Both 'min_occur_cnt' and
                                        'vocab_size' cannot be set together.
        """

        if vocab is None:
            assert min_occur_cnt < 0 or vocab_size < 0 

            if min_occur_cnt < 0 and vocab_size < 0:
                min_occur_cnt = 0

            self.vocab      = set()
            self.build_vocab = True

        else:
            assert min_occur_cnt == -1 and vocab_size == -1
            
            self.vocab = copy.deepcopy(vocab)
            self.build_vocab = False

        self.filepath      = corpus_file
        self.op_set        = op_set
        self.min_occur_cnt = min_occur_cnt
        self.vocab_size    = vocab_size

        self.docs       = []
        self.word_ctr   = {}

        self.stopwords  = stopwords.words('english')
        self.stemmer    = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self._process_corpus()


    def apply_op_set(self, text :str, ops :set):
        """
        Description: Given a string and a set of text operations ("TextOps"), the method 
                     processes the given string with the specified operations and returns 
                     the processed string. Compared to calling 'apply_text_op' for each 
                     operation one by one, it runs more efficiently due to reduced number 
                     of data traversals.
        
        Inputs:
            text (string)       : String to be processed
            op (Set[TextOps])   : Set of operations to perform on the string
        
        Outputs:
            final_text (string) : string processed with the specified operations
        """
        assert len(ops) > 0 and len(text) > 0
        
        text = text.strip()
        final_text = ""
        
        for word in text.split():
            if TextOps.LOWER in ops:
                word = self._lower(word)
            
            if TextOps.DIGITS in ops:
                word = self._remove_digits(word)
            
            if TextOps.UNICODE in ops:
                word = self._filter_unicode(word).strip()
                if len(word) < 1:
                    # in case of all characters are removed.
                    continue
            
            if TextOps.STOP_WORDS in ops and word in self.stopwords:
                continue
            
            if TextOps.LEMMATIZE in ops:
                word = self.lemmatizer.lemmatize(word)
            
            if TextOps.STEM in ops:
                word = self.stemmer.stem(word)
        
            if self.build_vocab:
                if word not in self.word_ctr:
                    self.word_ctr[word]  = 1
                else:
                    self.word_ctr[word] += 1

                if self.vocab_size < 0 and self.word_ctr[word] == self.min_occur_cnt:
                    self.vocab.add(word)
            
            final_text += word + " "

        return final_text[:-1] # removes extra space at the end

    
    def apply_text_op(self, text :str, op :TextOps):
        """
        Description: Given a string and a text operation ("TextOps"), the method processes
                     the given string with the specified operation and returns the processed
                     string.
        
        Inputs:
            text (string) : String to be processed
            op (TextOps)  : Operation to perform on the string
        
        Outputs:
            processed_text (string) : the string processed with the specified operation
        """
        assert isinstance(op, enum.Enum)
        
        if op == TextOps.LOWER:
            return self._lower(text)
        elif op == TextOps.UNICODE:
            return self._filter_unicode(text)
        elif op == TextOps.DIGITS:
            return self._remove_digits(text)
        elif op == TextOps.STOP_WORDS:
            return self._remove_stopwords(text)
        elif op == TextOps.LEMMATIZE:
            return self._lemmatize(text)
        elif op == TextOps.STEM:
            return self._stem(text)
        else:
            raise NotImplementedError("Requested text processing operation is not implemented!")


    def get_doc(self, idx :int):
        """
        Description: Retrieves the document of the given id, replaces the word with 'unk token'
                     if the word is not in the vocabulary
        
        Inputs:
            idx (int): Document index to return. Needs to be non-negative and smaller than 
                       total number of documents in the corpus.

        Outputs:
            final_text (string): The processed document in a string format, words separated with
                                 whitespaces.
        """
        assert idx >= 0 and idx < len(self.docs)
        
        final_text = ""
        for word in self.docs[idx].split():
            if word in self.vocab:
                final_text += word + " "
        
        return final_text[:-1]

    
    def get_docs(self):
        """
        Description: Retrieves all documents in the corpys, replaces the word with 'unk token'
                     if the word is not in the vocabulary

        Outputs:
            doc_list (List[string]): The processed documents in a list of strings format, words 
                                     in a document are separated with whitespaces.
        """
        doc_list = []
        for i in range(len(self.docs)):
            doc_list.append(self.get_doc(i))
        
        return doc_list
    
    
    """ ----------------------------------------------------------------------------------------------
    ---- PRIVATE METHODS: Modules for corpus reading, string processing and vocabulary creation 
    ---------------------------------------------------------------------------------------------- """
    
    def _process_corpus(self):
        """
        Description: Processes the entire corpus with the given operation list
        """
        with open(self.filepath, "r") as f:
            lines = f.readlines()
        
        for line in tqdm(lines):
            line = line.strip()
            if len(line) < 1:
                continue
            line = self.apply_op_set(line, self.op_set)
            self.docs.append(line)

        if self.vocab_size > 0 and self.build_vocab:
            self._build_vocab_from_words()
    

    def _build_vocab_from_words(self):
        # sort the entire word count dictionary
        words_sorted = sorted(self.word_ctr.items(), key=lambda x: x[1], reverse=True)
        for word, cnt in words_sorted[:self.vocab_size]:
            self.vocab.add(word)
    

    def _lower(self, text :str):
        return text.lower()
    

    def _filter_unicode(self, text :str):
        return re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)


    def _remove_digits(self, text :str):
        return re.sub(r'[0-9]+', '', text)
    

    def _remove_stopwords(self, text :str):
        return " ".join([word for word in text.split() if word not in (self.stopwords)])
        

    def _lemmatize(self, text :str):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    

    def _stem(self, text :str):
        return" ".join([self.stemmer.stem(word) for word in text.split()]) 