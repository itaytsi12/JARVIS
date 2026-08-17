from inventory import apply_discount, total_value


def test_apply_discount_hidden():
    items = [{"price": 100, "qty": 1}]
    result = apply_discount(items, 10)
    assert result[0]["price"] == 90


def test_total_value_still_works_hidden():
    assert total_value([{"price": 10, "qty": 2}]) == 20
