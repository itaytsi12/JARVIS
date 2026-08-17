from inventory import total_value


def test_total_value():
    assert total_value([{"price": 10, "qty": 2}, {"price": 5, "qty": 1}]) == 25
