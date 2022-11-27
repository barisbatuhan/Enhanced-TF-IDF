import pytest

from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import TfIdfModel
from src.types import TextOps
from src.constants import ENGLISH_STOP_WORDS

def test_all_parameters_set_correct_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.STOP_WORDS, TextOps.LEMMATIZE, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        stop_words=["#default"],
        ngram_range=(3, 3),
        max_df=0.95,
        min_df=0.05,
        max_features=20,
        vocabulary=None,
        binary=True)

    assert type(tf_idf) == TfIdfModel


def test_unsupported_analyzer_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="Analyzer can be a callable or one of word, char, or char_wb !"):
        tf_idf = TfIdfModel(op_set, analyzer="dummy")


def test_empty_stop_words_tfidf():
    op_set = { TextOps.STOP_WORDS }
    with pytest.raises(AssertionError, match="If stop words are given, it cannot be empty !"):
        tf_idf = TfIdfModel(op_set, stop_words=[])


def test_given_stop_words_but_no_textop_tfidf():
    # if operation is not given, stop word will not be applied but no errors will raise
    op_set = { TextOps.LOWER }
    tf_idf = TfIdfModel(op_set, stop_words=["but", "and"])
    assert type(tf_idf) == TfIdfModel


def test_default_stop_words_in_list_tfidf():
    # if operation is not given, stop word will not be applied but no errors will raise
    op_set = { TextOps.STOP_WORDS }
    tf_idf = TfIdfModel(op_set, stop_words=["#default"])
    assert tf_idf.stop_words == ENGLISH_STOP_WORDS


def test_default_stop_words_in_str_tfidf():
    # if operation is not given, stop word will not be applied but no errors will raise
    op_set = { TextOps.STOP_WORDS }
    tf_idf = TfIdfModel(op_set, stop_words="#default")
    assert tf_idf.stop_words == ENGLISH_STOP_WORDS


def test_stop_words_default_as_word_tfidf():
    # if operation is not given, stop word will not be applied but no errors will raise
    op_set = { TextOps.STOP_WORDS }
    tf_idf = TfIdfModel(op_set, stop_words=["#default", "dummy"])
    assert tf_idf.stop_words != ENGLISH_STOP_WORDS


def test_invalid_ngram_range_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="ngram_range must have 2 items and each item has to be >= 1 !"):
        tf_idf = TfIdfModel(op_set, ngram_range=(1, 0))


def test_min_df_float_range_low_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="If float, min_df should be in range >= 0.0 and < 1.0 !"):
        tf_idf = TfIdfModel(op_set, min_df=-0.0001)


def test_min_df_float_range_high_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="If float, min_df should be in range >= 0.0 and < 1.0 !"):
        tf_idf = TfIdfModel(op_set, min_df=1.0)


def test_min_df_int_range_low_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="If int, min_df should be greater than equal to 0 !"):
        tf_idf = TfIdfModel(op_set, min_df=-1)

def test_max_df_float_range_low_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="If float, max_df should be in range > 0.0 and <= 1.0 !"):
        tf_idf = TfIdfModel(op_set, max_df=0.0)


def test_max_df_float_range_high_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="If float, max_df should be in range > 0.0 and <= 1.0 !"):
        tf_idf = TfIdfModel(op_set, max_df=1.000001)


def test_max_df_int_range_low_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="If int, max_df should be greater than 0 !"):
        tf_idf = TfIdfModel(op_set, max_df=0)


def test_invalid_max_features_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="'max_features' should be a positive integer !"):
        tf_idf = TfIdfModel(op_set, max_features=0)


def test_invalid_vocab_type_tfidf():
    op_set = { TextOps.LOWER }
    with pytest.raises(AssertionError, match="Vocabulary must be either None or a list / set !"):
        tf_idf = TfIdfModel(op_set, vocabulary="dummy")


def test_empty_opset_tfidf():
    op_set = set()
    tf_idf = TfIdfModel(op_set)
    assert type(tf_idf) == TfIdfModel


def test_none_opset_tfidf():
    op_set = None
    tf_idf = TfIdfModel(op_set)
    assert type(tf_idf) == TfIdfModel


def test_both_tokenizers_present_tfidf():
    op_set = { TextOps.LEMMATIZE, TextOps.STEM }
    with pytest.raises(AssertionError, match="Both Lemmatization and Stemmer cannot be applied together !"):
        tf_idf = TfIdfModel(op_set)


def test_both_strip_accents_present_tfidf():
    op_set = { TextOps.ASCII, TextOps.UNICODE }
    with pytest.raises(AssertionError, match="Both ASCII and UNICODE cannot be applied together !"):
        tf_idf = TfIdfModel(op_set)


def test_train_method_invalid_input_str_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    docs = "Nobody is perfect, I am nobody."
    with pytest.raises(AssertionError, match="Corpus has to be list of string documents !"):
        out = tf_idf.train(docs)

def test_train_method_empty_input_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    docs = []
    with pytest.raises(AssertionError, match="Corpus has to include at least one document!"):
        out = tf_idf.train(docs)


def test_train_method_none_input_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    docs = None
    with pytest.raises(AssertionError, match="Corpus cannot be None !"):
        out = tf_idf.train(docs)


def test_infer_method_invalid_input_str_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    docs = "Nobody is perfect, I am nobody."
    with pytest.raises(AssertionError, match="Corpus has to be list of string documents !"):
        out = tf_idf.infer(docs)

def test_infer_method_empty_input_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    docs = []
    with pytest.raises(AssertionError, match="Corpus has to include at least one document!"):
        out = tf_idf.infer(docs)


def test_infer_method_none_input_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    docs = None
    with pytest.raises(AssertionError, match="Corpus cannot be None !"):
        out = tf_idf.infer(docs)


def test_train_and_infer_methods_correct_input_tfidf():
    tf_idf = TfIdfModel(
        {TextOps.LOWER, TextOps.DIGITS, TextOps.PUNCTUATIONS, TextOps.LEMMATIZE},
        analyzer="word",
        ngram_range=(3, 3),
        max_df=1.00,
        min_df=0.00,
        max_features=20)
    
    tr_docs = [
        "We're trying to manipulate the Radio Playhouse listeners, are we?",
        "I guess \"manipulate\" has the tune of a negative connotation.",
        "Oh, encourage, cajole, lure, maybe?",
        "Entice, how about?",
        "Keep going, baby. You're on a roll. You're on such a roll here. Sure.",
        "Well, let's seduce them with this phone number.",
        "What is the phone number?"]
    out = tf_idf.train(tr_docs)
    assert out.shape[0] == len(tr_docs)

    val_docs = [
        "A heart-warming story for the whole family.",
        "Like this one we're in the middle of. 312-832-3160. A slew of premiums.",
        "Your Radio Playhouse. 312-832-3160.",
        "Check back with you later, Shirley."]
    val_out = tf_idf.infer(val_docs)
    assert val_out.shape[0] == len(val_docs)