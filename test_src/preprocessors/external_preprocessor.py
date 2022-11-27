import pytest
from sklearn.feature_extraction.text import strip_accents_unicode, strip_accents_ascii

from src.preprocessors import PuncPreprocessor, ExternalPreprocessor

def test_delete_punctuations_external():
    init_str = "random_string_230498_20\"'''`#11ai-ic+&%/\n\t lalalalal"
    final_str = "random string 230498 20 11ai ic \n\t lalalalal"
    pp = PuncPreprocessor()
    ep = ExternalPreprocessor(pp)
    assert ep(init_str) == final_str


def test_no_punctuations_external():
    init_str = "randomstring"
    final_str = "randomstring"
    pp = PuncPreprocessor()
    ep = ExternalPreprocessor(pp)
    assert ep(init_str) == final_str


def test_nonascii_text_with_punctuations_external():
    init_str = "ışöçğ654787%+(&)(&865+-*/"
    final_str = "ışöçğ654787 865"
    pp = PuncPreprocessor()
    ep = ExternalPreprocessor(pp)
    assert ep(init_str) == final_str


def test_empty_text_external():
    init_str = ""
    final_str = ""
    pp = PuncPreprocessor()
    ep = ExternalPreprocessor(pp)
    assert ep(init_str) == final_str


def test_non_fn_arg_external():
    non_fn = "string value"
    with pytest.raises(AssertionError, match="Passed argument is not a callable!"):
        ep = ExternalPreprocessor(non_fn)


def test_ascii_strip_external():
    init_str = "āăąēîïĩíĝġńñšŝśûůŷşöçğ"
    final_str = "aaaeiiiiggnnsssuuysocg"
    ep = ExternalPreprocessor(strip_accents_ascii)
    assert ep(init_str) == final_str


def test_unicode_strip_external():
    init_str = "āăąēîïĩíĝġńñšŝśûůŷşöçğ"
    final_str = "aaaeiiiiggnnsssuuysocg"
    ep = ExternalPreprocessor(strip_accents_unicode)
    assert ep(init_str) == final_str