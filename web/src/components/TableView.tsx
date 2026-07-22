import { useState } from "react";
import type { Product } from "../types";

type SortKey = "price" | "rating" | "review_count";

export default function TableView({ products }: { products: Product[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("price");
  const [ascending, setAscending] = useState(true);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setAscending(!ascending);
    } else {
      setSortKey(key);
      setAscending(true);
    }
  }

  const sorted = [...products].sort((a, b) => {
    const av = a[sortKey] ?? (ascending ? Infinity : -Infinity);
    const bv = b[sortKey] ?? (ascending ? Infinity : -Infinity);
    return ascending ? av - bv : bv - av;
  });

  return (
    <table className="product-table">
      <thead>
        <tr>
          <th>Title</th>
          <th onClick={() => handleSort("price")}>Price</th>
          <th onClick={() => handleSort("rating")}>Rating</th>
          <th onClick={() => handleSort("review_count")}>Reviews</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((p, i) => (
          <tr key={i}>
            <td>{p.url ? <a href={p.url} target="_blank" rel="noreferrer">{p.title}</a> : p.title}</td>
            <td>{p.price != null ? `$${p.price.toFixed(2)}` : "-"}</td>
            <td>{p.rating != null ? p.rating : "-"}</td>
            <td>{p.review_count ?? "-"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
