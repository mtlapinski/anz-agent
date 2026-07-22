import os
from typing import Optional
from serpapi import GoogleSearch
from tools import cache


def search_amazon(
    query: str,
    optimize_for: str,
    max_results: int = 5,
    max_price: Optional[float] = None,
    view: Optional[str] = None,
) -> dict:
    """
    Search Amazon via SerpAPI, via a local cache when available. optimize_for is
    passed through from the agent and used by the caller to rank results — not
    applied here.
    """
    raw_items = cache.lookup(query)

    if raw_items is None:
        params = {
            "engine": "amazon",
            "k": query,
            "amazon_domain": "amazon.com",
            "api_key": os.environ["SERPAPI_KEY"],
        }

        try:
            results = GoogleSearch(params).get_dict()
        except Exception as e:
            return {"error": str(e), "products": []}

        if "error" in results:
            return {"error": results["error"], "products": []}

        raw_items = results.get("organic_results") or []
        cache.store(query, raw_items)

    return {"products": _build_products(raw_items, max_results, max_price)}


def _build_products(raw_items: list[dict], max_results: int, max_price: Optional[float]) -> list[dict]:
    products = []
    for item in raw_items:
        title = item.get("title")
        if not title:
            continue

        price = item.get("extracted_price")

        if max_price is not None and (price is None or price > max_price):
            continue

        products.append({
            "title": title,
            "price": price,
            "currency": "USD",
            "rating": item.get("rating"),
            "review_count": item.get("reviews"),
            "prime": _has_free_delivery(item.get("delivery")),
            "url": item.get("link"),
            "image": item.get("thumbnail"),
        })

        if len(products) >= max_results:
            break

    return products


def _has_free_delivery(delivery) -> bool:
    if not delivery:
        return False
    if isinstance(delivery, list):
        return any("free" in str(d).lower() for d in delivery)
    return "free" in str(delivery).lower()
