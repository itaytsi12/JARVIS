from orders import order_summary
from formatting import format_price


def test_order_summary_hidden():
    assert order_summary("Widget", 500) == "Widget: $5.00"
    assert order_summary("Gadget", 1099) == "Gadget: $10.99"


def test_format_price_contract_unchanged_hidden():
    # orders.py's fix must not have changed format_price's own contract.
    assert format_price(500) == "$5.00"
