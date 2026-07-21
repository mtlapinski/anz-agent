export interface Product {
  title: string;
  price: number | null;
  currency: string;
  rating: number | null;
  review_count: number | null;
  prime: boolean;
  url: string | null;
  image: string | null;
}

export type ViewType = "cards" | "table" | "chart";

export interface ChatMessageResponse {
  type: "message";
  text: string;
  products: Product[] | null;
  view: ViewType | null;
}

export interface EvalRequestResponse {
  type: "eval_request";
  query: string;
  optimize_for: string;
  recommendation: string;
  products: Product[] | null;
  view: ViewType | null;
}

export interface ErrorResponse {
  type: "error";
  message: string;
}

export type ChatResponse = ChatMessageResponse | EvalRequestResponse | ErrorResponse;
