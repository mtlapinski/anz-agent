# Web UI Design System & Layout Fixes

## Context

The web UI (added in [2026-07-20-web-ui-design.md](2026-07-20-web-ui-design.md)) is functionally complete but visually unstyled: chat replies render as one unformatted text blob (no markdown), and the results panel has a layout bug where long product titles get clipped instead of wrapping. There is no consistent visual language across the chat pane, results panel, session bar, and eval widget — each was styled ad hoc during implementation.

This spec addresses styling and layout only. No behavior, API contract, or component structure changes.

## Goals

- A small, consistent design system (CSS custom properties) covering color, spacing, and radius, so restyling is a one-place edit rather than hunting through per-component CSS.
- Full markdown rendering in the chat pane (bold, lists, links, paragraphs) for assistant/system messages.
- Fix the product-card title truncation bug (root cause: missing `min-width: 0` on a flex child) — titles wrap onto multiple lines instead of clipping.
- A cohesive look across chat bubbles, results panel (cards/table/chart), session bar, and eval widget.

## Non-goals

- No new build tooling (no Tailwind, no component library) — plain CSS with custom properties, consistent with the existing lightweight approach from the original web UI implementation.
- No dark mode in this pass — light mode only.
- No grid layout for cards — stays single-column (per explicit preference), just fixes the wrapping bug and applies consistent styling.
- No new frontend test framework — verification stays manual in-browser, consistent with the existing project convention.
- No changes to component structure, state management, or the FastAPI backend.

## Design Tokens

A `:root` block in `web/src/index.css` defines the palette and scale:

```css
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
```

Every component's CSS references these variables instead of hardcoded values. These initial values are a starting point, not final — expected to be tuned visually once built (tracked as a known follow-up, not a blocker for this spec).

## Chat Pane

- **Markdown rendering**: add `react-markdown` as a dependency. Assistant and system message text renders through `<ReactMarkdown>` instead of being dropped in as a raw string, so bold, bulleted/numbered lists, links, and paragraph breaks render correctly. User messages remain plain text (no markdown parsing needed for what the user typed).
- **Chat bubbles**:
  - User messages: right-aligned, `--color-user-bubble` background, white text.
  - Assistant messages: left-aligned, `--color-assistant-bubble` background, `--color-text`.
  - System messages (errors, rating acknowledgment): centered, small text, `--color-system-text`, no bubble background — visually distinct from conversational turns.
  - All bubbles: `--radius-lg`, `--space-3` padding, max-width ~75% of the chat pane so long messages wrap rather than stretching full width.
- **Eval widget**: restyled as a distinct card (`--color-surface` background, `--color-border` outline, `--radius-md`) rather than a chat bubble, so it reads as an interactive control, not a conversational message.

## Results Panel

**Root cause of the current truncation bug**: `.product-card` is a flex row (image + text column). The text column has no `min-width: 0`, so flexbox's default `min-width: auto` keeps it from shrinking below its content's natural width — long titles overflow the card and get clipped by the browser instead of wrapping.

**Fix + restyle:**
- `.product-card`'s text column gets `min-width: 0` and `flex: 1`.
- Product titles get `overflow-wrap: break-word` so long titles wrap onto multiple lines; the card grows taller as needed (no information lost, no ellipsis).
- Cards remain single-column (stacked vertically), each styled with `--color-border` outline, `--radius-md`, `--space-3` padding, `--color-surface` background.
- `TableView` and `ChartView` receive the same token-based styling (border colors, header font-weight, hover states on sortable table headers) so all three view types read as one visual system.
- The "No search results yet." empty state gets centered, muted (`--color-text-muted`) styling instead of unstyled plain text.

## Session Bar & Overall Layout

- **Session bar** (provider/model picker shown before the first message): restyled as a centered card (`--color-surface`, `--color-border`, `--radius-lg`) rather than an inline flex row — it's a one-time setup screen, warranting more visual weight than a persistent toolbar.
- **Two-pane layout**: add a header label above each pane ("Chat" / "Results") for orientation. The existing divider between panes uses `--color-border`.
- **Typography**: tighten line-height for body text readability; use `--font-weight-medium` for pane headers and card titles to distinguish them from body text.

## Testing

No new frontend test framework (per existing project constraint). Manual in-browser verification covers:
- Chat markdown renders correctly (bold, lists, links) in assistant replies.
- Long product titles wrap without clipping in `CardsView`.
- Chat bubble alignment and colors match the spec (user right/blue, assistant left/gray, system centered/muted).
- `TableView` and `ChartView` share the same visual language as `CardsView` (borders, spacing, typography).
- Session bar and empty-state read as intentionally designed, not unstyled.
