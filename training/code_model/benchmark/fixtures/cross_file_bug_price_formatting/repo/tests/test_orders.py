from orders import order_summary


def test_order_summary():
    assert order_summary("Widget", 500) == "Widget: $5.00"
