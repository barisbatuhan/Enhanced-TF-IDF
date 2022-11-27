import pytest

from src.preprocessors import DigitPreprocessor

def test_delete_digits():
    init_str = "random_string_230498_20#11ai-ic+&%/\n\t lalalalal"
    final_str = "random_string__#ai-ic+&%/\n\t lalalalal"
    dp = DigitPreprocessor()
    assert dp(init_str) == final_str


def test_no_digits():
    init_str = "random_string"
    final_str = "random_string"
    dp = DigitPreprocessor()
    assert dp(init_str) == final_str


def test_multi_spaces_digits():
    init_str = "random 1244 string"
    final_str = "random string"
    dp = DigitPreprocessor()
    assert dp(init_str) == final_str


def test_nonascii_text_with_digits():
    init_str = "ışöçğ654787%+(&)(&865+-*/"
    final_str = "ışöçğ%+(&)(&+-*/"
    dp = DigitPreprocessor()
    assert dp(init_str) == final_str


def test_empty_text():
    init_str = ""
    final_str = ""
    dp = DigitPreprocessor()
    assert dp(init_str) == final_str


def test_none_text():
    init_str = None
    dp = DigitPreprocessor()
    with pytest.raises(AssertionError, match="Text to preprocess cannot be None"):
        dp(init_str)


def test_no_string_text():
    init_str = 2394293.596
    dp = DigitPreprocessor()
    with pytest.raises(TypeError):
        dp(init_str)