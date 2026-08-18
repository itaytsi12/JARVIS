class PricingService:
    """Coordinates ProductRepository (source of truth) and ProductCache
    (read-through cache) so callers get fast reads without ever seeing a
    stale price."""

    def __init__(self, repository, cache):
        self.repository = repository
        self.cache = cache

    def update_price(self, product_id, new_price) -> None:
        self.repository.update_price(product_id, new_price)
        # BUG: the repository is updated, but the cache is never told the
        # price changed -- get_current_price keeps returning the stale
        # cached value for this product until the process restarts.

    def get_current_price(self, product_id):
        return self.cache.get_price(product_id)
