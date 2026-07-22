import type { Product } from "../types";

export default function CardsView({ products }: { products: Product[] }) {
  return (
    <div>
      {products.map((p, i) => (
        <div className="product-card" key={i}>
          {p.image && <img src={p.image} alt={p.title} />}
          <div className="product-card-text">
            <div className="product-card-title">{p.title}</div>
            <div className="product-card-meta">
              <span>{p.price != null ? `$${p.price.toFixed(2)}` : "Price unavailable"}</span>
              <span>{p.rating != null ? `${p.rating}★ (${p.review_count ?? 0})` : "No rating"}</span>
              {p.prime && <span>Prime</span>}
            </div>
            {p.url && <a href={p.url} target="_blank" rel="noreferrer" className="product-card-link">View on Amazon</a>}
          </div>
        </div>
      ))}
    </div>
  );
}
