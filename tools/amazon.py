import os
from typing import Optional
from amazon_paapi import AmazonApi


def search_amazon(
    query: str,
    optimize_for: str,
    max_results: int = 5,
    max_price: Optional[float] = None,
) -> dict:
    amazon = AmazonApi(
        key=os.environ["AMAZON_ACCESS_KEY"],
        secret=os.environ["AMAZON_SECRET_KEY"],
        tag=os.environ["AMAZON_PARTNER_TAG"],
        country="US",
    )

    try:
        items = amazon.search_items(keywords=query, item_count=max_results * 2)
    except Exception as e:
        return {"error": str(e), "products": []}

    products = []
    for item in items:
        try:
            price = item.offers.listings[0].price.amount
        except (AttributeError, IndexError):
            price = None

        if max_price is not None and price is not None and price > max_price:
            continue

        try:
            currency = item.offers.listings[0].price.currency
        except (AttributeError, IndexError):
            currency = "USD"

        try:
            rating = item.customer_reviews.star_rating.value
        except AttributeError:
            rating = None

        try:
            review_count = item.customer_reviews.count.display_value
        except AttributeError:
            review_count = 0

        try:
            prime = item.offers.listings[0].delivery_info.is_prime_eligible
        except (AttributeError, IndexError):
            prime = False

        try:
            title = item.item_info.title.display_value
        except AttributeError:
            continue  # skip items with no title

        products.append({
            "title": title,
            "price": price,
            "currency": currency,
            "rating": rating,
            "review_count": review_count,
            "prime": prime,
            "url": item.detail_page_url,
        })

        if len(products) >= max_results:
            break

    return {"products": products}
