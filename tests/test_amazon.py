import json
import os
import pytest
from unittest.mock import patch, MagicMock


def make_mock_item(
    title="Test Laptop",
    price=499.99,
    currency="USD",
    rating=4.5,
    review_count=1234,
    prime=True,
    url="https://www.amazon.com/dp/B001TEST",
):
    item = MagicMock()
    item.item_info.title.display_value = title
    item.offers.listings[0].price.amount = price
    item.offers.listings[0].price.currency = currency
    item.customer_reviews.star_rating.value = rating
    item.customer_reviews.count.display_value = review_count
    item.offers.listings[0].delivery_info.is_prime_eligible = prime
    item.detail_page_url = url
    return item


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_returns_products(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = [make_mock_item()]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert "products" in result
    assert len(result["products"]) == 1
    p = result["products"][0]
    assert p["title"] == "Test Laptop"
    assert p["price"] == 499.99
    assert p["currency"] == "USD"
    assert p["rating"] == 4.5
    assert p["review_count"] == 1234
    assert p["prime"] is True
    assert p["url"] == "https://www.amazon.com/dp/B001TEST"


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_filters_by_max_price(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = [
        make_mock_item(title="Cheap Laptop", price=199.99),
        make_mock_item(title="Expensive Laptop", price=999.99),
    ]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5, max_price=300.0)

    assert len(result["products"]) == 1
    assert result["products"][0]["title"] == "Cheap Laptop"


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_handles_no_results(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = []

    from tools.amazon import search_amazon
    result = search_amazon(query="xyznonexistent", optimize_for="price", max_results=5)

    assert result["products"] == []


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_handles_api_error(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.side_effect = Exception("Rate limit exceeded")

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert "error" in result
    assert "Rate limit exceeded" in result["error"]
    assert result["products"] == []


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_respects_max_results(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = [make_mock_item(title=f"Item {i}") for i in range(10)]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=3)

    assert len(result["products"]) == 3


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_handles_none_response(mock_api_class):
    mock_api = mock_api_class.return_value
    mock_api.search_items.return_value = None

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert result["products"] == []


@patch.dict(os.environ, {
    "AMAZON_ACCESS_KEY": "fake_key",
    "AMAZON_SECRET_KEY": "fake_secret",
    "AMAZON_PARTNER_TAG": "fake_tag",
})
@patch("tools.amazon.AmazonApi")
def test_search_excludes_unpriced_items_when_max_price_set(mock_api_class):
    mock_api = mock_api_class.return_value
    priced_item = make_mock_item(title="Priced Laptop", price=199.99)
    unpriced_item = MagicMock()
    unpriced_item.item_info.title.display_value = "Unpriced Laptop"
    unpriced_item.offers.listings[0].price.amount = None
    unpriced_item.detail_page_url = "https://www.amazon.com/dp/B002TEST"
    mock_api.search_items.return_value = [priced_item, unpriced_item]

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5, max_price=300.0)

    titles = [p["title"] for p in result["products"]]
    assert "Priced Laptop" in titles
    assert "Unpriced Laptop" not in titles
