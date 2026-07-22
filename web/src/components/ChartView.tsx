import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { Product } from "../types";

interface ChartPoint {
  title: string;
  price: number;
  rating: number;
}

export default function ChartView({ products }: { products: Product[] }) {
  const data: ChartPoint[] = products
    .filter((p): p is Product & { price: number; rating: number } => p.price != null && p.rating != null)
    .map((p) => ({ price: p.price, rating: p.rating, title: p.title }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid />
        <XAxis type="number" dataKey="price" name="Price" unit="$" />
        <YAxis type="number" dataKey="rating" name="Rating" domain={[0, 5]} />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          content={({ active, payload }) => {
            if (!active || !payload || !payload.length) return null;
            const p = payload[0].payload as ChartPoint;
            return (
              <div style={{ background: "var(--color-bg)", border: "1px solid var(--color-border)", padding: "var(--space-2)", borderRadius: "var(--radius-sm)" }}>
                <div>{p.title}</div>
                <div>${p.price.toFixed(2)} — {p.rating}★</div>
              </div>
            );
          }}
        />
        <Scatter data={data} fill="var(--color-accent)" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
