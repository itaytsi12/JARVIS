from calc import add


def test_add_hidden():
    assert add(10, -3) == 7
    assert add(0, 0) == 0
    assert add(-5, -5) == -10
