import type { Product } from "../types";

export default function CardsView({ products }: { products: Product[] }) {
  return (
    <div>
      {products.map((p, i) => (
        <div className="product-card" key={i}>
          {p.image && <img src={p.image} alt={p.title} />}
          <div>
            <div><strong>{p.title}</strong></div>
            <div>{p.price != null ? `$${p.price.toFixed(2)}` : "Price unavailable"}</div>
            <div>{p.rating != null ? `${p.rating}★ (${p.review_count ?? 0})` : "No rating"}</div>
            <div>{p.prime ? "Prime" : ""}</div>
            {p.url && <a href={p.url} target="_blank" rel="noreferrer">View on Amazon</a>}
          </div>
        </div>
      ))}
    </div>
  );
}
