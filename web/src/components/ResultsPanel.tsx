import type { Product, ViewType } from "../types";
import CardsView from "./CardsView";
import TableView from "./TableView";
import ChartView from "./ChartView";

interface Props {
  products: Product[] | null;
  view: ViewType | null;
}

export default function ResultsPanel({ products, view }: Props) {
  if (!products || products.length === 0) {
    return (
      <div className="results-panel">
        <div className="pane-header">Results</div>
        <div className="results-empty">No search results yet.</div>
      </div>
    );
  }

  const effectiveView = view ?? "cards";

  return (
    <div className="results-panel">
      <div className="pane-header">Results</div>
      {effectiveView === "table" && <TableView products={products} />}
      {effectiveView === "chart" && <ChartView products={products} />}
      {effectiveView === "cards" && <CardsView products={products} />}
    </div>
  );
}
