from domain import Product
from repository import ProductRepository
from cache import ProductCache
from service import PricingService


def _make_service():
    repo = ProductRepository()
    repo.save(Product("p1", "Widget", 10.0))
    repo.save(Product("p2", "Gadget", 20.0))
    cache = ProductCache(repo)
    service = PricingService(repo, cache)
    return service, cache


def test_price_update_is_reflected_immediately_hidden():
    service, _ = _make_service()
    assert service.get_current_price("p1") == 10.0
    service.update_price("p1", 15.0)
    assert service.get_current_price("p1") == 15.0


def test_multiple_updates_are_all_reflected_hidden():
    service, _ = _make_service()
    service.update_price("p1", 15.0)
    assert service.get_current_price("p1") == 15.0
    service.update_price("p1", 12.5)
    assert service.get_current_price("p1") == 12.5


def test_caching_is_preserved_for_unrelated_products_hidden():
    """Catches the naive 'symptom fix' of bypassing the cache entirely --
    an unrelated product's price must still only be fetched from the
    repository ONCE, even after several reads."""
    service, cache = _make_service()
    service.get_current_price("p2")
    service.get_current_price("p2")
    service.get_current_price("p2")
    assert cache.fetch_count == 1
