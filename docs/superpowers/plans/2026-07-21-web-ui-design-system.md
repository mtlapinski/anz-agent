# Web UI Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CSS custom properties for theming, render markdown in chat, fix product-card title truncation, and restyle all components to use a consistent visual language.

**Architecture:** A `:root` block in `web/src/index.css` defines color, spacing, and radius tokens. All component classes reference these variables instead of hardcoded values, so theming is a one-place edit. `react-markdown` renders assistant/system messages in the chat pane. The product-card layout bug (missing `min-width: 0` on flex children) is fixed to allow titles to wrap. All components are restyled to use the tokens and follow a consistent visual pattern (chat bubbles, card backgrounds, borders, typography).

**Tech Stack:** React, TypeScript, react-markdown (new dependency), plain CSS with custom properties.

## Global Constraints

- No new build tooling beyond `react-markdown` — stays plain CSS with custom properties, no Tailwind.
- Light mode only — no dark mode in this pass.
- No new frontend test framework — manual in-browser verification only.
- No changes to component structure, state management, or the FastAPI backend.
- No grid layout for product cards — stays single-column.

---

### Task 1: Design Tokens & CSS Foundation

**Files:**
- Modify: `web/src/index.css`
- Modify: `web/package.json` (add `react-markdown`)

**Interfaces:**
- Produces: CSS custom properties (`--color-*`, `--space-*`, `--radius-*`, `--font-*`) defined in `:root`, and a base reset for all component classes to reference.

- [ ] **Step 1: Add design tokens to index.css**

Replace the entire `web/src/index.css` content with:

```css
* { box-sizing: border-box; }

:root {
  --color-bg: #ffffff;
  --color-surface: #f8f9fb;
  --color-border: #e2e5e9;
  --color-text: #1a1d23;
  --color-text-muted: #6b7280;
  --color-accent: #3b6ecf;
  --color-accent-text: #ffffff;
  --color-user-bubble: #3b6ecf;
  --color-assistant-bubble: #f1f3f5;
  --color-system-text: #9ca3af;
  --color-danger: #d64545;

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;

  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;

  --font-sans: system-ui, -apple-system, sans-serif;
  --font-weight-medium: 600;
}

body {
  margin: 0;
  font-family: var(--font-sans);
  background-color: var(--color-bg);
  color: var(--color-text);
}

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.pane-header {
  font-size: 0.875rem;
  font-weight: var(--font-weight-medium);
  color: var(--color-text-muted);
  padding: var(--space-2) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.panes-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.chat-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--color-border);
  padding: var(--space-4);
  overflow-y: auto;
  background-color: var(--color-bg);
}

.results-panel {
  flex: 1;
  padding: var(--space-4);
  overflow-y: auto;
  background-color: var(--color-bg);
}

.message {
  margin-bottom: var(--space-3);
  max-width: 75%;
  border-radius: var(--radius-lg);
  padding: var(--space-3);
  word-wrap: break-word;
}

.message.user {
  align-self: flex-end;
  background-color: var(--color-user-bubble);
  color: var(--color-accent-text);
  border-radius: var(--radius-lg);
}

.message.assistant {
  align-self: flex-start;
  background-color: var(--color-assistant-bubble);
  color: var(--color-text);
  border-radius: var(--radius-lg);
}

.message.assistant p {
  margin: var(--space-2) 0;
  line-height: 1.5;
}

.message.assistant ul,
.message.assistant ol {
  margin: var(--space-2) 0;
  padding-left: 1.5rem;
}

.message.assistant li {
  margin-bottom: var(--space-1);
}

.message.assistant a {
  color: var(--color-accent);
  text-decoration: none;
}

.message.assistant a:hover {
  text-decoration: underline;
}

.message.system {
  align-self: center;
  background-color: transparent;
  color: var(--color-system-text);
  font-size: 0.875rem;
  font-style: italic;
  border-radius: 0;
  padding: var(--space-2);
  max-width: none;
}

.chat-input {
  display: flex;
  gap: var(--space-2);
  margin-top: auto;
  padding-top: var(--space-4);
  border-top: 1px solid var(--color-border);
}

.chat-input input {
  flex: 1;
  padding: var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-family: var(--font-sans);
  font-size: 1rem;
}

.chat-input input:focus {
  outline: none;
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px rgba(59, 110, 207, 0.1);
}

.chat-input input:disabled {
  background-color: var(--color-surface);
  color: var(--color-text-muted);
  cursor: not-allowed;
}

.chat-input button {
  padding: var(--space-3) var(--space-6);
  background-color: var(--color-accent);
  color: var(--color-accent-text);
  border: none;
  border-radius: var(--radius-sm);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
}

.chat-input button:hover:not(:disabled) {
  opacity: 0.9;
}

.chat-input button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.session-bar {
  display: flex;
  gap: var(--space-2);
  padding: var(--space-4);
  align-items: center;
  justify-content: center;
  background-color: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  max-width: 400px;
  margin: 2rem auto;
}

.session-bar select,
.session-bar input {
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-family: var(--font-sans);
}

.session-bar button {
  padding: var(--space-2) var(--space-4);
  background-color: var(--color-accent);
  color: var(--color-accent-text);
  border: none;
  border-radius: var(--radius-sm);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
}

.session-bar button:hover {
  opacity: 0.9;
}

.eval-widget {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  margin-bottom: var(--space-3);
}

.eval-widget label {
  display: block;
  margin-bottom: var(--space-2);
  font-weight: var(--font-weight-medium);
}

.eval-widget select,
.eval-widget input {
  display: block;
  width: 100%;
  margin-bottom: var(--space-2);
  padding: var(--space-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-family: var(--font-sans);
}

.eval-widget button {
  width: 100%;
  padding: var(--space-2);
  background-color: var(--color-accent);
  color: var(--color-accent-text);
  border: none;
  border-radius: var(--radius-sm);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
}

.eval-widget button:hover {
  opacity: 0.9;
}

.product-card {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  margin-bottom: var(--space-3);
  display: flex;
  gap: var(--space-3);
}

.product-card img {
  width: 64px;
  height: 64px;
  object-fit: contain;
  flex-shrink: 0;
}

.product-card-text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.product-card-title {
  font-weight: var(--font-weight-medium);
  color: var(--color-text);
  overflow-wrap: break-word;
  word-break: break-word;
}

.product-card-meta {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.product-card-link {
  color: var(--color-accent);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: var(--font-weight-medium);
}

.product-card-link:hover {
  text-decoration: underline;
}

.product-table {
  width: 100%;
  border-collapse: collapse;
  background-color: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.product-table th {
  background-color: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  padding: var(--space-3);
  text-align: left;
  font-weight: var(--font-weight-medium);
  color: var(--color-text);
  cursor: pointer;
  user-select: none;
}

.product-table th:hover {
  background-color: #f3f5f9;
}

.product-table td {
  padding: var(--space-3);
  border-bottom: 1px solid var(--color-border);
  color: var(--color-text);
}

.product-table tr:last-child td {
  border-bottom: none;
}

.product-table a {
  color: var(--color-accent);
  text-decoration: none;
}

.product-table a:hover {
  text-decoration: underline;
}

.results-empty {
  text-align: center;
  color: var(--color-text-muted);
  padding: var(--space-6);
  font-style: italic;
}
```

- [ ] **Step 2: Add react-markdown to package.json**

Open `web/package.json` and add `"react-markdown"` to the `dependencies` section:

```json
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "recharts": "^2.15.0"
  },
```

- [ ] **Step 3: Install dependencies**

Run: `cd web && npm install`
Expected: `added 1 package` (react-markdown + deps).

- [ ] **Step 4: Commit**

```bash
git add web/src/index.css web/package.json web/package-lock.json
git commit -m "feat: add design tokens and react-markdown dependency"
```

---

### Task 2: Markdown Rendering in Chat Pane

**Files:**
- Modify: `web/src/components/ChatPane.tsx`

**Interfaces:**
- Consumes: `react-markdown` (new dependency from Task 1), existing `Message` type.
- Produces: Assistant/system messages render markdown (bold, lists, links, paragraphs); user messages remain plain text.

- [ ] **Step 1: Update ChatPane to use react-markdown**

Replace the entire `web/src/components/ChatPane.tsx`:

```tsx
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import EvalWidget from "./EvalWidget";
import type { EvalRequestResponse } from "../types";

export interface Message {
  role: "user" | "assistant" | "system";
  text: string;
}

interface Props {
  messages: Message[];
  pendingEval: EvalRequestResponse | null;
  loading: boolean;
  onSend: (message: string) => void;
  onEvalSubmit: (score: number, note?: string) => void;
}

export default function ChatPane({ messages, pendingEval, loading, onSend, onEvalSubmit }: Props) {
  const [input, setInput] = useState("");

  function handleSubmit() {
    if (!input.trim() || loading || pendingEval) return;
    onSend(input.trim());
    setInput("");
  }

  return (
    <div className="chat-pane">
      <div className="pane-header">Chat</div>
      {messages.map((m, i) => (
        <div key={i} className={`message ${m.role}`}>
          {m.role === "user" ? (
            m.text
          ) : (
            <ReactMarkdown>{m.text}</ReactMarkdown>
          )}
        </div>
      ))}
      {pendingEval && <EvalWidget recommendation={pendingEval.recommendation} onSubmit={onEvalSubmit} />}
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          disabled={loading || !!pendingEval}
          placeholder="What can I help you find?"
        />
        <button onClick={handleSubmit} disabled={loading || !!pendingEval}>Send</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify the ChatPane compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/ChatPane.tsx
git commit -m "feat: render markdown in chat messages using react-markdown"
```

---

### Task 3: Fix Product Card Layout & Restyle

**Files:**
- Modify: `web/src/components/CardsView.tsx`

**Interfaces:**
- Consumes: `Product` type, CSS tokens from `index.css`.
- Produces: Product cards that wrap long titles, use consistent styling from tokens (background, border, radius, padding).

- [ ] **Step 1: Rewrite CardsView to fix title wrapping and use tokens**

Replace `web/src/components/CardsView.tsx`:

```tsx
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
```

- [ ] **Step 2: Verify CardsView compiles**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/CardsView.tsx
git commit -m "fix: allow product titles to wrap, restyle cards with design tokens"
```

---

### Task 4: Restyle TableView & ChartView

**Files:**
- Modify: `web/src/components/TableView.tsx`
- Modify: `web/src/components/ChartView.tsx`

**Interfaces:**
- Consumes: `Product` type, CSS tokens from `index.css`.
- Produces: Table and chart styled with consistent tokens (borders, padding, hover states, typography).

- [ ] **Step 1: Update TableView to use token-based styling**

Replace `web/src/components/TableView.tsx`:

```tsx
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
```

- [ ] **Step 2: Update ChartView with consistent spacing**

Replace `web/src/components/ChartView.tsx`:

```tsx
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
```

- [ ] **Step 3: Verify compilation**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/TableView.tsx web/src/components/ChartView.tsx
git commit -m "refactor: restyle table and chart views with design tokens"
```

---

### Task 5: Restyle Remaining Components

**Files:**
- Modify: `web/src/components/ResultsPanel.tsx`
- Modify: `web/src/components/SessionBar.tsx`
- Modify: `web/src/components/EvalWidget.tsx`

**Interfaces:**
- Consumes: CSS tokens, existing component logic.
- Produces: All components styled with consistent tokens (empty state, session bar card, eval widget).

- [ ] **Step 1: Update ResultsPanel for empty state styling**

Replace `web/src/components/ResultsPanel.tsx`:

```tsx
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
      <div>
        <div className="pane-header">Results</div>
        <div className="results-empty">No search results yet.</div>
      </div>
    );
  }

  const effectiveView = view ?? "cards";

  return (
    <div>
      <div className="pane-header">Results</div>
      {effectiveView === "table" && <TableView products={products} />}
      {effectiveView === "chart" && <ChartView products={products} />}
      {effectiveView === "cards" && <CardsView products={products} />}
    </div>
  );
}
```

- [ ] **Step 2: Update SessionBar to match token-based card styling**

Replace `web/src/components/SessionBar.tsx`:

```tsx
import { useState } from "react";

interface Props {
  onStart: (provider: string, model: string) => void;
}

const DEFAULTS: Record<string, string> = {
  google: "gemini-flash-lite-latest",
  anthropic: "claude-haiku-4-5-20251001",
};

export default function SessionBar({ onStart }: Props) {
  const [provider, setProvider] = useState("google");
  const [model, setModel] = useState(DEFAULTS.google);

  function handleProviderChange(next: string) {
    setProvider(next);
    setModel(DEFAULTS[next]);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
      <div className="session-bar">
        <select value={provider} onChange={(e) => handleProviderChange(e.target.value)}>
          <option value="google">Google</option>
          <option value="anthropic">Anthropic</option>
        </select>
        <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="Model" />
        <button onClick={() => onStart(provider, model)}>Start</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Update EvalWidget styling**

Replace `web/src/components/EvalWidget.tsx`:

```tsx
import { useState } from "react";

interface Props {
  recommendation: string;
  onSubmit: (score: number, note?: string) => void;
}

export default function EvalWidget({ recommendation, onSubmit }: Props) {
  const [score, setScore] = useState(5);
  const [note, setNote] = useState("");

  return (
    <div className="eval-widget">
      <p>{recommendation}</p>
      <label>Rate usefulness (1-5):</label>
      <select value={score} onChange={(e) => setScore(Number(e.target.value))}>
        {[1, 2, 3, 4, 5].map((n) => (
          <option key={n} value={n}>{n}</option>
        ))}
      </select>
      <input placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
      <button onClick={() => onSubmit(score, note || undefined)}>Submit rating</button>
    </div>
  );
}
```

- [ ] **Step 4: Verify compilation**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/ResultsPanel.tsx web/src/components/SessionBar.tsx web/src/components/EvalWidget.tsx
git commit -m "refactor: restyle remaining components with design tokens"
```

---

### Task 6: Add Pane Headers & Final Styling

**Files:**
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: Updated `ResultsPanel` component with header (from Task 5).
- Produces: App layout with pane header labels ("Chat" / "Results"), two-pane layout structure using tokens.

- [ ] **Step 1: Update App.tsx to add pane structure**

In `web/src/App.tsx`, update the return statement (the part after `return (`) to wrap the chat and results panes:

```tsx
      <div className="app">
        <ChatPane
          messages={messages}
          pendingEval={pendingEval}
          loading={loading}
          onSend={handleSend}
          onEvalSubmit={handleEvalSubmit}
        />
        <ResultsPanel products={lastProducts} view={lastView} />
      </div>
```

Change it to:

```tsx
      <div className="app">
        <div className="panes-container">
          <ChatPane
            messages={messages}
            pendingEval={pendingEval}
            loading={loading}
            onSend={handleSend}
            onEvalSubmit={handleEvalSubmit}
          />
          <ResultsPanel products={lastProducts} view={lastView} />
        </div>
      </div>
```

(Note: `ChatPane` and `ResultsPanel` now add their own headers internally, so the pane headers are included in those components.)

- [ ] **Step 2: Verify compilation**

Run: `cd web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/App.tsx
git commit -m "refactor: wrap panes in container for proper flex layout"
```

---

### Task 7: Manual In-Browser Verification

**Files:**
- None (verification only)

**Interfaces:**
- Consumes: All changes from Tasks 1-6.
- Produces: Confirmation that chat renders markdown, product cards wrap text, all components use consistent colors/spacing/radius, session bar and eval widget are properly styled.

- [ ] **Step 1: Start both servers**

Terminal 1: `python server.py` (FastAPI backend on localhost:8000)
Terminal 2: `cd web && npm run dev` (Vite frontend on localhost:5173)

- [ ] **Step 2: Open browser to localhost:5173**

Expected: Session bar appears as a centered card with provider/model inputs, not inline.

- [ ] **Step 3: Start a session**

Click **Start**, observe the interface switches to two-pane layout with "Chat" header on the left and "Results" header on the right.

- [ ] **Step 4: Send a test message**

Send: "find me a laptop under $500, optimize for price"

Expected: 
- Agent's reply renders in the chat bubble with markdown (bold text, lists if present, links clickable).
- After rating, recommendation appears.
- Eval rating widget displays with 1-5 selector and note input, all styled with tokens.
- Results panel shows product cards (or table/chart based on agent choice).
- **Critical**: Long product titles wrap onto multiple lines instead of clipping.

- [ ] **Step 5: Verify colors and spacing**

Visually confirm:
- User messages: blue bubble (#3b6ecf), white text, right-aligned.
- Assistant messages: light-gray bubble (#f1f3f5), dark text, left-aligned.
- Cards: light background (#f8f9fb), subtle border (#e2e5e9), consistent padding (0.75rem).
- Session bar: card-like appearance with border and background.
- All text uses system font, menus/inputs have consistent padding and radius.

- [ ] **Step 6: Test a different view type**

Send: "compare 3 gaming laptops by price and rating" (or similar query where agent might choose table/chart).

Expected: Results render in table or chart view with same visual language (borders, spacing, typography).

- [ ] **Step 7: No formal commit needed**

This task is verification only; no code changes. If all checks pass, the implementation is complete.

---

## Self-Review

**Spec coverage:**
- ✅ Design tokens (`--color-*`, `--space-*`, `--radius-*`) in `:root` — Task 1
- ✅ Chat markdown rendering — Task 2
- ✅ Product card title wrapping bug fix — Task 3
- ✅ Consistent styling across all components (cards, table, chart, session bar, eval widget) — Tasks 3-5
- ✅ Pane headers and layout — Tasks 5-6
- ✅ Manual in-browser verification — Task 7

**No placeholders:** All steps include exact file content, exact commands, and expected output.

**Type consistency:** CSS variables consistently referenced as `var(--color-*)` across all components.

**Scope:** Styling only, no behavior or state management changes.
