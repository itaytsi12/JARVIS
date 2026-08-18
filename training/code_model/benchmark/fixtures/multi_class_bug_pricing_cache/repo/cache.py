class ProductCache:
    """A simple read-through price cache in front of a ProductRepository.
    `fetch_count` lets tests verify the repository isn't hit for a product
    whose cached price is still valid -- caching must keep working for
    products that were never touched by a price update."""

    def __init__(self, repository):
        self.repository = repository
        self._cache = {}
        self.fetch_count = 0

    def get_price(self, product_id):
        if product_id not in self._cache:
            product = self.repository.get(product_id)
            self._cache[product_id] = product.price if product else None
            self.fetch_count += 1
        return self._cache[product_id]

    def invalidate(self, product_id) -> None:
        self._cache.pop(product_id, None)
