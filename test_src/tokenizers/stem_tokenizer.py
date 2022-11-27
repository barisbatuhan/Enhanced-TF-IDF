import pytest

from src.tokenizers import StemTokenizer 


def test_correct_stem():
    init_str = "changes does filming ordered they toys"
    final_list = ['chang', 'doe', 'film', 'order', 'they', 'toy']
    st = StemTokenizer()
    assert st(init_str) == final_list


def test_non_str_float_input_stem():
    init_str = 124918.4198
    st = StemTokenizer()
    with pytest.raises(TypeError):
        st(init_str)


def test_non_str_list_input_stem():
    init_list = ["abc", "defg"]
    st = StemTokenizer()
    with pytest.raises(TypeError):
        st(init_list)


def test_non_word_str_input_stem():
    # even if a word is unknown, -s is removed by stemmer
    init_str = "ierfeys nfefeo oupeefv"
    final_list = ["ierfey", "nfefeo", "oupeefv"]
    st = StemTokenizer()
    assert st(init_str) == final_list


def test_none_input_stem():
    init_str = None
    st = StemTokenizer()
    with pytest.raises(AssertionError, match="Text to tokenize cannot be None or empty !"):
        st(init_str)


def test_empty_input_stem():
    init_str = ""
    st = StemTokenizer()
    with pytest.raises(AssertionError, match="Text to tokenize cannot be None or empty !"):
        st(init_str)


def test_empty_tokenizer_fn_stem():
    init_str = "changes does filming ordered they toys"
    st = StemTokenizer()
    st.tokenizer_fn = None
    with pytest.raises(AssertionError, match="Tokenizer function must be a callable !"):
        st(init_str)