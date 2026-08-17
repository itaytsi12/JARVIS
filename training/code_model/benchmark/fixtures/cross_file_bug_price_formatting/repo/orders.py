from formatting import format_price


def order_summary(item_name, price_cents):
    return f"{item_name}: {format_price(price_cents * 100)}"
