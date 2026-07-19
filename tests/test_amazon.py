import os
import pytest
from unittest.mock import patch, MagicMock


def make_mock_result(
    title="Test Laptop",
    extracted_price=499.99,
    rating=4.5,
    reviews=1234,
    delivery="FREE delivery",
    link="https://www.amazon.com/dp/B001TEST",
):
    return {
        "title": title,
        "extracted_price": extracted_price,
        "rating": rating,
        "reviews": reviews,
        "delivery": delivery,
        "link": link,
    }


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_returns_products(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {
        "organic_results": [make_mock_result()]
    }

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


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_filters_by_max_price(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {
        "organic_results": [
            make_mock_result(title="Cheap Laptop", extracted_price=199.99),
            make_mock_result(title="Expensive Laptop", extracted_price=999.99),
        ]
    }

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5, max_price=300.0)

    assert len(result["products"]) == 1
    assert result["products"][0]["title"] == "Cheap Laptop"


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_handles_no_results(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {"organic_results": []}

    from tools.amazon import search_amazon
    result = search_amazon(query="xyznonexistent", optimize_for="price", max_results=5)

    assert result["products"] == []


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_handles_api_error(mock_search_class):
    mock_search_class.return_value.get_dict.side_effect = Exception("Rate limit exceeded")

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert "error" in result
    assert "Rate limit exceeded" in result["error"]
    assert result["products"] == []


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_respects_max_results(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {
        "organic_results": [make_mock_result(title=f"Item {i}") for i in range(10)]
    }

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=3)

    assert len(result["products"]) == 3


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_handles_missing_organic_results(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {}

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5)

    assert result["products"] == []


@patch.dict(os.environ, {"SERPAPI_KEY": "fake_key"})
@patch("tools.amazon.GoogleSearch")
def test_search_excludes_unpriced_items_when_max_price_set(mock_search_class):
    mock_search_class.return_value.get_dict.return_value = {
        "organic_results": [
            make_mock_result(title="Priced Laptop", extracted_price=199.99),
            make_mock_result(title="Unpriced Laptop", extracted_price=None),
        ]
    }

    from tools.amazon import search_amazon
    result = search_amazon(query="laptop", optimize_for="price", max_results=5, max_price=300.0)

    titles = [p["title"] for p in result["products"]]
    assert "Priced Laptop" in titles
    assert "Unpriced Laptop" not in titles
