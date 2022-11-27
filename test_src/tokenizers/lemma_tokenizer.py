import pytest

from src.tokenizers import LemmaTokenizer 


def test_correct_lemma():
    init_str = "changes does filming ordered they toys"
    final_list = ['change', 'doe', 'filming', 'ordered', 'they', 'toy']
    lt = LemmaTokenizer()
    assert lt(init_str) == final_list


def test_non_str_float_input_lemma():
    init_str = 124918.4198
    lt = LemmaTokenizer()
    with pytest.raises(TypeError):
        lt(init_str)


def test_non_str_list_input_lemma():
    init_list = ["abc", "defg"]
    lt = LemmaTokenizer()
    with pytest.raises(TypeError):
        lt(init_list)


def test_non_word_str_input_lemma():
    init_str = "ierfeys nfefeo oupeefv"
    final_list = ["ierfeys", "nfefeo", "oupeefv"]
    lt = LemmaTokenizer()
    assert lt(init_str) == final_list


def test_none_input_lemma():
    init_str = None
    lt = LemmaTokenizer()
    with pytest.raises(AssertionError, match="Text to tokenize cannot be None or empty !"):
        lt(init_str)


def test_empty_input_lemma():
    init_str = ""
    lt = LemmaTokenizer()
    with pytest.raises(AssertionError, match="Text to tokenize cannot be None or empty !"):
        lt(init_str)


def test_empty_tokenizer_fn_lemma():
    init_str = "changes does filming ordered they toys"
    lt = LemmaTokenizer()
    lt.tokenizer_fn = None
    with pytest.raises(AssertionError, match="Tokenizer function must be a callable !"):
        lt(init_str)