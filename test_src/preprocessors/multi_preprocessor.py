import pytest

from src.preprocessors import DigitPreprocessor, PuncPreprocessor, ExternalPreprocessor, MultiPreprocessor

def test_no_list_but_callable_multi():
    with pytest.raises(AssertionError, match="Preprocessor list is not a list !"):
        mp = MultiPreprocessor(DigitPreprocessor())


def test_no_list_no_callable_multi():
    non_fn = "basic string"
    with pytest.raises(AssertionError, match="Preprocessor list is not a list !"):
        mp = MultiPreprocessor(non_fn)


def test_no_callable_in_list_multi():
    callables = [PuncPreprocessor(), "no_callable", DigitPreprocessor()]
    with pytest.raises(AssertionError, match="Parameter passed as preprocessor is not a callable !"):
        mp = MultiPreprocessor(callables)
    

def test_all_callable_in_list_multi():
    init_str = "random_string_230498_20\"'''`#11ai-ic+&%/\n\t lalalalal"
    final_str = "random string ai ic \n\t lalalalal"
    callables = [PuncPreprocessor(), DigitPreprocessor()]
    mp = MultiPreprocessor(callables)
    assert mp(init_str) == final_str


def test_all_callable_in_list_reverse_order_multi():
    init_str = "random_string_230498_20\"'''`#11ai-ic+&%/\n\t lalalalal"
    final_str = "random string ai ic \n\t lalalalal"
    callables = [DigitPreprocessor(), PuncPreprocessor()]
    mp = MultiPreprocessor(callables)
    assert mp(init_str) == final_str