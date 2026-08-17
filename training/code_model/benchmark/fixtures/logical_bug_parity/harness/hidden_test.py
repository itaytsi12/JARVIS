from parity import is_even


def test_is_even_hidden():
    assert is_even(2)
    assert is_even(0)
    assert not is_even(3)
    assert not is_even(-1)
