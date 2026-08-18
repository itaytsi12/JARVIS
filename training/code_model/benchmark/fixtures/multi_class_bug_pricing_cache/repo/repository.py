from domain import Product


class ProductRepository:
    """The source of truth for product records. Deliberately correct and
    simple -- the bug is not here."""

    def __init__(self):
        self._products = {}

    def save(self, product: Product) -> None:
        self._products[product.product_id] = product

    def get(self, product_id):
        return self._products.get(product_id)

    def update_price(self, product_id, new_price) -> None:
        product = self._products.get(product_id)
        if product is not None:
            product.price = new_price
