def total_value(items):
    return sum(item["price"] * item["qty"] for item in items)


def apply_discount(items, percent):
    for item in items:
        item["price"] = item["price"]  # BUG: discount never applied
    return items
