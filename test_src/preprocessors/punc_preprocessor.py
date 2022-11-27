import pytest

from src.preprocessors import PuncPreprocessor

def test_delete_punctuations():
    init_str = "random_string_230498_20\"'''`#11ai-ic+&%/\n\t lalalalal"
    final_str = "random string 230498 20 11ai ic \n\t lalalalal"
    pp = PuncPreprocessor()
    assert pp(init_str) == final_str


def test_no_punctuations():
    init_str = "randomstring"
    final_str = "randomstring"
    pp = PuncPreprocessor()
    assert pp(init_str) == final_str


def test_nonascii_text_with_punctuations():
    init_str = "ışöçğ654787%+(&)(&865+-*/"
    final_str = "ışöçğ654787 865"
    pp = PuncPreprocessor()
    assert pp(init_str) == final_str


def test_empty_text():
    init_str = ""
    final_str = ""
    pp = PuncPreprocessor()
    assert pp(init_str) == final_str


def test_none_text():
    init_str = None
    pp = PuncPreprocessor()

    with pytest.raises(AssertionError, match="Text to preprocess cannot be None"):
        pp(init_str)


def test_no_string_text():
    init_str = 2394293.596
    pp = PuncPreprocessor()

    with pytest.raises(TypeError):
        pp(init_str)