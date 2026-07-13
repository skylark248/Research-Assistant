# Phase 8 Design: UI/UX Refresh

Date: 2026-07-13
Status: approved

## Goal

The phase-4 web UI works but is bare (system-default styling, no dark mode)
and carries known interaction debt: agent-activity lines splatter between
messages and land under the streaming reply, errors surface as inline text,
there are no empty/loading states, and the layout is unusable on mobile.
Phase 8 is a visual refresh plus those UX fixes in one coherent pass.

Direction (locked): **clean research tool** — quiet neutral palette + one
accent, tuned system font stack, information-dense spacing, subtle 1px
borders over shadows, auto dark mode with a manual toggle.

## Non-goals

- No framework, no build step, no external fonts/CDNs — the vanilla-JS,
  offline-capable stance stands (vendored `marked` + `DOMPurify` untouched).
- No API/SSE protocol changes; no Python changes beyond none-at-all.
- No new features beyond stated UX fixes (no timestamps, no message editing,
  no search, no settings panel).
- No JS test harness (established stance) — verification is a live browser
  smoke checklist.

## Decisions (locked)

- **Activity display**: collapsible block per reply (accordion), not a
  status strip or side panel.
- **Dark mode**: `prefers-color-scheme` default, manual toggle overrides,
  persisted in `localStorage`, applied as `data-theme` on `<html>`.
- **Mobile**: sidebar collapses behind a hamburger below 720px.
- **Ingest** moves from the top-of-page fieldset into a compact collapsible
  section at the sidebar bottom.

## Architecture

### Files

| File | Change |
|---|---|
| `api/static/styles.css` | NEW — all styling, extracted from index.html's `<style>`, token-based |
| `api/static/index.html` | restructured layout: header bar, sidebar, chat column, input row; `<link>` to styles.css; tiny inline script only for pre-paint theme application |
| `api/static/app.js` | restructured around the turn component; same SSE parsing, same endpoints |
| `README.md` | phase-8 row + UI feature bullets refresh |

### 1. Design system (styles.css)

- `:root` custom properties: `--bg`, `--bg-raised`, `--fg`, `--fg-muted`,
  `--border`, `--accent`, `--accent-fg`, `--warn-bg`, `--warn-fg`,
  `--radius: 8px`, spacing scale `--s1..--s6` (4/8/12/16/24/32px).
- Dark theme = same variable names redefined under
  `html[data-theme="dark"]`; a `@media (prefers-color-scheme: dark)` block
  applies the dark values when `data-theme` is absent (i.e. "auto").
- Font: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
  `code`/`pre` get the mono stack. Base 14px/1.55.
- Theme boot: 3-line inline script in `<head>` reads
  `localStorage.theme` and stamps `data-theme` before first paint (no
  flash). Toggle button cycles auto → light → dark (title shows current).

### 2. Layout (index.html)

- **Header bar**: app title, provider `<select>`, theme toggle button,
  hamburger (mobile only).
- **Sidebar** (left, 260px): "New conversation" button, thread list,
  collapsible "Ingest papers" section at the bottom (`<details>` element —
  free accordion, no JS).
- **Chat column**: centered, `max-width: 760px`; scrollable log; input row
  pinned at the bottom (textarea + send button).
- **Mobile (<720px)**: sidebar becomes an off-canvas drawer toggled by the
  hamburger; backdrop click closes it; selecting a thread closes it.

### 3. Turn component (app.js)

Each exchange is one `.turn` element built by `startTurn(userText)`:

```
.turn
  .msg.user            — user bubble
  details.activity     — accordion; <summary> shows live status / final count
    .activity-lines    — one line per SSE status event
  .msg.assistant       — streamed text, markdown-rendered on done
  .meta-row            — citation chips + "⚠ citations unverified" badge + copy button
```

- SSE routing: `status` events append into the CURRENT turn's
  `.activity-lines` (accordion auto-opens on first line); `delta` streams
  into the assistant bubble; `turn_end` with tools moves interim reasoning
  text into the activity block (today's behavior, relocated); `done`
  markdown-renders the reply, fills the meta row, collapses the accordion
  and sets its summary to `⚙ N steps`; `error` marks the turn failed.
- The accordion stays attached to its turn permanently — activity history
  is preserved per reply and never interleaves with later messages.
- Copy button: `navigator.clipboard.writeText` of the reply's raw markdown
  text (kept on the turn object); brief "✓ copied" feedback.
- Thread restore (`openThread`) renders turns without activity blocks
  (transcripts carry none — existing contract, unchanged).

### 4. States

- **Empty log** (new conversation): welcome card — one-line blurb + 3
  example questions as buttons; clicking sends that question. Card is
  removed on first send.
- **No providers available** (`/api/providers` returns none available):
  amber banner above the input, textarea + send disabled, hint text
  ("start Ollama or add an API key to .env").
- **Errors**: toast component (top-right, auto-dismiss 6s, dismiss on
  click) replaces every inline "Chat failed:" / "Ingest failed:" status
  line; a failed turn's assistant bubble shows "⚠ failed — try again".
- **Loading**: ingest button disabled + spinner glyph while in flight;
  thread restore shows a brief "loading…" placeholder row.
- **Scroll pinning**: auto-scroll only when the log is already at the
  bottom (within 40px); otherwise a floating "↓ new messages" pill appears;
  clicking it scrolls down and resumes pinning.
- **Input**: `<textarea>` replaces the text input; auto-grows 1–6 lines;
  Enter sends, Shift+Enter inserts newline; send button disabled while a
  stream is in flight (double-send guard preserved).

### 5. Untouched

All API endpoints and payloads, SSE event names/shapes, thread persistence,
`marked`/`DOMPurify` vendoring, every Python file, all tests.

## Error handling

| Failure | Behavior |
|---|---|
| SSE `error` event | toast + turn marked failed; input re-enabled |
| fetch throws (network) | same toast path |
| `/api/providers` all unavailable | banner + disabled input (recheck on page load only) |
| transcript 404 | toast "No transcript for this thread" (replaces silent no-op) |
| clipboard API unavailable (http) | copy button hidden |

## Testing / verification

- `uv run pytest` stays green (no server changes; `test_index_served`
  checks title text — keep "Paper Research Assistant" in the header).
- Live browser smoke checklist (executed at the end of the phase, Qdrant +
  Ollama up): stream a question end-to-end; activity accordion streams,
  collapses to step count, re-expands; unverified badge renders on
  `faithful=false`; dark/light/auto toggle persists across reload; mobile
  viewport (375px) drawer works; empty-state card sends an example
  question; kill Qdrant → error toast (not inline text); thread restore
  renders turns + citations; copy button copies markdown; scroll-up during
  a stream shows the pill instead of yanking.
- README: phase-8 table row; web-UI bullet list updated (dark mode,
  activity accordion, mobile drawer, toasts).
