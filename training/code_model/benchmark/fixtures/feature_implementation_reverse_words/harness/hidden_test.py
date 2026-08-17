from strings_util import reverse_words


def test_reverse_words_hidden():
    assert reverse_words("hello world") == "world hello"
    assert reverse_words("a b c") == "c b a"
    assert reverse_words("single") == "single"
