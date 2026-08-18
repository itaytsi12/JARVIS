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


def test_get_current_price_returns_initial_price():
    service, _ = _make_service()
    assert service.get_current_price("p1") == 10.0
