# Phase 8: UI/UX Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Visual refresh (token design system, dark mode, real layout) plus interaction fixes (per-reply activity accordion, empty/loading/error states, scroll pinning, textarea input, mobile drawer) for the vanilla-JS web UI — zero backend changes.

**Architecture:** Styling extracted from index.html into token-based `api/static/styles.css` (light+dark via CSS custom properties). index.html restructured (header / sidebar / chat column) with tiny inline scripts owning theme + drawer. app.js rebuilt around a `.turn` component that keeps each reply's agent activity in a collapsible `<details>` attached to that reply. Spec: `docs/superpowers/specs/2026-07-13-phase-8-ui-ux-refresh-design.md`.

**Tech Stack:** Vanilla JS + CSS, vendored `marked`/`DOMPurify` (untouched), FastAPI static serving (untouched). No build step, no CDNs, no new dependencies.

## Global Constraints

- Frontend-only: NO changes to any Python file, API endpoint, SSE event name/shape, or vendored lib.
- No framework, no build step, no external fonts/CDNs (offline-capable stance).
- `test_index_served` asserts the text "Paper Research Assistant" — it must remain in the page (header `<h1>`).
- Element IDs consumed by app.js are fixed: `log`, `thread-list`, `provider`, `chat-input`, `chat-btn`, `new-conv-btn`, `ingest-query`, `ingest-btn`, `ingest-status` (Task 1 preserves them so the old app.js keeps working between tasks); new IDs introduced here: `topbar`, `menu-btn`, `theme-btn`, `backdrop`, `shell`, `sidebar`, `main`, `ingest`, `scroll-pill`, `provider-banner`, `chat-row`, `toasts`.
- Theme: `prefers-color-scheme` is the default; manual override cycles auto → light → dark, stored in `localStorage.theme`, applied as `data-theme` on `<html>`; a pre-paint inline script prevents flash.
- Mobile breakpoint: 720px; sidebar becomes an off-canvas drawer.
- Double-send guard (`inFlight`) must be preserved.
- Verification: `uv run pytest` green after every task (no server changes may break it); full browser smoke checklist runs at the end of the phase (services required), not per task.
- Commit style: imperative conventional (`feat:`/`docs:`), matching `git log`.

---

### Task 1: Design system + layout shell (`styles.css` + `index.html`)

**Files:**
- Create: `api/static/styles.css`
- Rewrite: `api/static/index.html`

**Interfaces:**
- Consumes: current `api/static/app.js` (unchanged this task — the preserved IDs keep it functional inside the new shell; its old inline styling classes get a small legacy CSS block, deleted in Task 3).
- Produces: all CSS classes Tasks 2–3 use: `.turn`, `.msg.user`, `.msg.assistant`, `.msg.failed`, `details.activity`, `.activity-line`, `.meta-row`, `.chip`, `.chip.warn`, `.copy-btn`, `.welcome`, `.welcome-title`, `.welcome-sub`, `.example`, `.toast`, `#scroll-pill`, `#provider-banner`, `.busy`; theme + drawer behavior owned by index.html inline scripts.

- [ ] **Step 1: Write `api/static/styles.css`**

```css
/* Phase 8 design system — tokens first, both themes defined once. */

:root {
  --bg: #fafafa;
  --bg-raised: #ffffff;
  --bg-inset: #f0f0f2;
  --fg: #1a1a20;
  --fg-muted: #6b6b76;
  --border: #e2e2e8;
  --accent: #4f46e5;
  --accent-fg: #ffffff;
  --warn-bg: #fff3cd;
  --warn-fg: #8a6d1a;
  --fail-fg: #b3261e;
  --radius: 8px;
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

html[data-theme="dark"] {
  --bg: #131318;
  --bg-raised: #1c1c24;
  --bg-inset: #23232d;
  --fg: #e6e6ec;
  --fg-muted: #9a9aa8;
  --border: #2e2e3a;
  --accent: #818cf8;
  --accent-fg: #131318;
  --warn-bg: #3a3320;
  --warn-fg: #e5c569;
  --fail-fg: #f2b8b5;
}

@media (prefers-color-scheme: dark) {
  html:not([data-theme]) {
    --bg: #131318;
    --bg-raised: #1c1c24;
    --bg-inset: #23232d;
    --fg: #e6e6ec;
    --fg-muted: #9a9aa8;
    --border: #2e2e3a;
    --accent: #818cf8;
    --accent-fg: #131318;
    --warn-bg: #3a3320;
    --warn-fg: #e5c569;
    --fail-fg: #f2b8b5;
  }
}

/* ---------- base ---------- */
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; font-family: var(--font); font-size: 14px; line-height: 1.55;
  background: var(--bg); color: var(--fg);
  display: flex; flex-direction: column;
}
code, pre { font-family: var(--mono); font-size: .92em; }
pre { background: var(--bg-inset); padding: var(--s3); border-radius: var(--radius); overflow-x: auto; }
button { font: inherit; color: inherit; background: var(--bg-raised); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--s2) var(--s3); cursor: pointer; }
button:hover { border-color: var(--fg-muted); }
button:disabled { opacity: .5; cursor: default; }
input[type="text"], textarea, select {
  font: inherit; color: var(--fg); background: var(--bg-raised);
  border: 1px solid var(--border); border-radius: var(--radius); padding: var(--s2) var(--s3);
}
textarea { resize: none; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

/* ---------- header ---------- */
#topbar {
  display: flex; align-items: center; gap: var(--s3);
  padding: var(--s2) var(--s4); border-bottom: 1px solid var(--border);
  background: var(--bg-raised);
}
#topbar h1 { font-size: 15px; font-weight: 600; margin: 0; flex: 1; }
#menu-btn { display: none; }
#theme-btn { padding: var(--s2); min-width: 36px; }

/* ---------- shell ---------- */
#shell { flex: 1; display: flex; min-height: 0; }

#sidebar {
  width: 260px; flex-shrink: 0; display: flex; flex-direction: column;
  border-right: 1px solid var(--border); background: var(--bg-raised);
  padding: var(--s3); gap: var(--s2); overflow-y: auto;
}
#sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--fg-muted); margin: var(--s2) 0 0; }
#new-conv-btn { width: 100%; }
#thread-list { flex: 1; min-height: 40px; }
.thread {
  display: flex; align-items: center; gap: var(--s1);
  padding: var(--s2) var(--s2); border-radius: var(--radius); cursor: pointer; font-size: 13px;
}
.thread:hover { background: var(--bg-inset); }
.thread.active { background: var(--bg-inset); font-weight: 600; }
.thread .title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.thread .del { border: none; background: none; color: var(--fg-muted); padding: 0 var(--s1); }
.thread .del:hover { color: var(--fail-fg); }
#ingest { border-top: 1px solid var(--border); padding-top: var(--s3); }
#ingest summary { cursor: pointer; color: var(--fg-muted); font-size: 13px; }
#ingest input { width: 100%; margin: var(--s2) 0; }
#ingest button { width: 100%; }

/* ---------- chat column ---------- */
#main { flex: 1; display: flex; flex-direction: column; align-items: center; min-width: 0; position: relative; }
#log { flex: 1; overflow-y: auto; width: 100%; max-width: 760px; padding: var(--s4); }
#chat-row { display: flex; gap: var(--s2); width: 100%; max-width: 760px; padding: var(--s3) var(--s4) var(--s4); align-items: flex-end; }
#chat-input { flex: 1; max-height: 148px; }
#chat-btn { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }

/* ---------- turn component ---------- */
.turn { margin-bottom: var(--s5); }
.msg { border-radius: var(--radius); padding: var(--s3) var(--s4); }
.msg.user {
  background: var(--accent); color: var(--accent-fg);
  margin-left: 20%; margin-bottom: var(--s2); width: fit-content; max-width: 80%;
  margin-left: auto;
}
.msg.assistant { background: var(--bg-raised); border: 1px solid var(--border); }
.msg.assistant p:first-child { margin-top: 0; }
.msg.assistant p:last-child { margin-bottom: 0; }
.msg.failed { color: var(--fail-fg); border-color: var(--fail-fg); }

details.activity {
  margin: 0 0 var(--s2); border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-inset); font-size: 12.5px; color: var(--fg-muted);
}
details.activity summary { cursor: pointer; padding: var(--s2) var(--s3); user-select: none; }
.activity-lines { padding: 0 var(--s3) var(--s2); display: flex; flex-direction: column; gap: 2px; }
.activity-line { font-family: var(--mono); font-size: 12px; white-space: pre-wrap; }

.meta-row { display: flex; flex-wrap: wrap; gap: var(--s1); align-items: center; margin-top: var(--s2); }
.chip {
  display: inline-block; font-size: 12px; background: var(--bg-inset); color: var(--fg-muted);
  border-radius: 10px; padding: 2px var(--s2); text-decoration: none;
}
a.chip:hover { color: var(--accent); }
.chip.warn { background: var(--warn-bg); color: var(--warn-fg); }
.copy-btn { font-size: 12px; padding: 2px var(--s2); margin-left: auto; border: none; background: none; color: var(--fg-muted); }
.copy-btn:hover { color: var(--fg); }

/* ---------- states ---------- */
.welcome { border: 1px dashed var(--border); border-radius: var(--radius); padding: var(--s5); text-align: center; margin-top: var(--s6); }
.welcome-title { font-weight: 600; margin-bottom: var(--s1); }
.welcome-sub { color: var(--fg-muted); font-size: 13px; margin-bottom: var(--s4); }
.example { display: block; width: 100%; text-align: left; margin-top: var(--s2); font-size: 13px; }

#scroll-pill {
  position: absolute; bottom: 84px; left: 50%; transform: translateX(-50%);
  background: var(--accent); color: var(--accent-fg); border: none;
  border-radius: 999px; padding: var(--s1) var(--s3); font-size: 12px;
}

#provider-banner {
  width: 100%; max-width: 760px; margin: 0 var(--s4); padding: var(--s2) var(--s3);
  background: var(--warn-bg); color: var(--warn-fg);
  border-radius: var(--radius); font-size: 13px;
}

#toasts { position: fixed; top: var(--s4); right: var(--s4); display: flex; flex-direction: column; gap: var(--s2); z-index: 30; }
.toast {
  background: var(--bg-raised); border: 1px solid var(--fail-fg); color: var(--fail-fg);
  border-radius: var(--radius); padding: var(--s3) var(--s4); max-width: 320px;
  font-size: 13px; cursor: pointer;
}

.busy::after { content: "…"; display: inline-block; animation: busy-dots 1s steps(4) infinite; }
@keyframes busy-dots { 0% { clip-path: inset(0 100% 0 0); } 100% { clip-path: inset(0 -8px 0 0); } }

.status { color: var(--fg-muted); font-size: 12.5px; margin: var(--s1) 0; }

/* ---------- mobile ---------- */
#backdrop { display: none; }
@media (max-width: 720px) {
  #menu-btn { display: block; }
  #sidebar {
    position: fixed; top: 0; bottom: 0; left: 0; z-index: 20;
    transform: translateX(-100%); transition: transform .18s ease;
  }
  body.sidebar-open #sidebar { transform: translateX(0); }
  body.sidebar-open #backdrop {
    display: block; position: fixed; inset: 0; z-index: 10; background: rgba(0,0,0,.4);
  }
  .msg.user { max-width: 92%; }
}

/* ---------- LEGACY classes for pre-Task-2 app.js — DELETE in Task 3 ---------- */
.user { color: var(--fg); background: var(--bg-inset); border-radius: var(--radius); padding: var(--s2) var(--s3); margin: var(--s2) 0; }
.bot { background: var(--bg-raised); border: 1px solid var(--border); border-radius: var(--radius); padding: var(--s2) var(--s3); margin: var(--s2) 0; }
.activity { color: var(--fg-muted); font-size: 12px; margin: 2px 0 2px var(--s4); }
.citations { margin: var(--s1) 0 var(--s3); }
.citations a { display: inline-block; font-size: 12px; background: var(--bg-inset); border-radius: 10px; padding: 2px var(--s2); margin-right: var(--s1); color: var(--fg-muted); text-decoration: none; }
.verdict { display: inline-block; font-size: 12px; background: var(--warn-bg); color: var(--warn-fg); border-radius: 10px; padding: 2px var(--s2); margin: var(--s1) 0 var(--s3); }
```

- [ ] **Step 2: Rewrite `api/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Research Assistant</title>
  <link rel="stylesheet" href="/styles.css">
  <script>
    // pre-paint theme stamp — prevents a light flash for dark-mode users
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      document.documentElement.dataset.theme = savedTheme;
    }
  </script>
</head>
<body>
  <header id="topbar">
    <button id="menu-btn" aria-label="Toggle sidebar">☰</button>
    <h1>Paper Research Assistant</h1>
    <select id="provider" title="Reasoning model"></select>
    <button id="theme-btn" aria-label="Toggle theme"></button>
  </header>

  <div id="backdrop"></div>

  <div id="shell">
    <nav id="sidebar">
      <button id="new-conv-btn">+ New conversation</button>
      <h2>Threads</h2>
      <div id="thread-list"></div>
      <details id="ingest">
        <summary>Ingest papers</summary>
        <input id="ingest-query" type="text" placeholder="arXiv search, e.g. attention is all you need">
        <button id="ingest-btn">Ingest</button>
        <div id="ingest-status" class="status"></div>
      </details>
    </nav>

    <main id="main">
      <div id="log"></div>
      <button id="scroll-pill" hidden>↓ new messages</button>
      <div id="provider-banner" hidden>
        No provider available — start Ollama (<code>ollama serve</code>) or add an
        API key to <code>.env</code>, then reload.
      </div>
      <div id="chat-row">
        <textarea id="chat-input" rows="1" placeholder="Ask about the ingested papers…"></textarea>
        <button id="chat-btn">Send</button>
      </div>
    </main>
  </div>

  <div id="toasts"></div>

  <script>
    // Theme toggle: auto → light → dark. Owned here, not by app.js.
    (function () {
      const btn = document.getElementById("theme-btn");
      function label() {
        const t = document.documentElement.dataset.theme || "auto";
        btn.textContent = t === "dark" ? "🌙" : t === "light" ? "☀️" : "🌓";
        btn.title = "theme: " + t;
      }
      btn.addEventListener("click", () => {
        const cur = document.documentElement.dataset.theme || "auto";
        const next = cur === "auto" ? "light" : cur === "light" ? "dark" : "auto";
        if (next === "auto") {
          localStorage.removeItem("theme");
          delete document.documentElement.dataset.theme;
        } else {
          localStorage.setItem("theme", next);
          document.documentElement.dataset.theme = next;
        }
        label();
      });
      label();
    })();
    // Mobile drawer. app.js closes it on thread selection via body class.
    (function () {
      document.getElementById("menu-btn").addEventListener("click",
        () => document.body.classList.toggle("sidebar-open"));
      document.getElementById("backdrop").addEventListener("click",
        () => document.body.classList.remove("sidebar-open"));
    })();
  </script>
  <script src="/vendor/marked.min.js"></script>
  <script src="/vendor/purify.min.js"></script>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Verify suite + interim behavior**

Run: `uv run pytest tests/test_api.py -q`
Expected: pass (`test_index_served` finds "Paper Research Assistant" in the `<h1>`).

Note: the OLD app.js still drives chat inside the new shell (IDs preserved; legacy CSS block keeps it presentable). Enter-to-send temporarily fires on Shift+Enter too — accepted between Tasks 1→3.

- [ ] **Step 4: Commit**

```bash
git add api/static/styles.css api/static/index.html
git commit -m "feat: token design system, dark mode, and layout shell for the web UI"
```

---

### Task 2: Turn component in `app.js`

**Files:**
- Rewrite: `api/static/app.js`

**Interfaces:**
- Consumes: Task 1's DOM (IDs above) + CSS classes; existing SSE protocol (`status`/`delta`/`turn_end`/`done`/`error`); existing endpoints.
- Produces: `startTurn(userText) -> turn`, `addActivityLine(turn, text)`, `finishActivity(turn)`, `fillMeta(turn, citations, faithful)`, `renderStaticTurn(userText, assistantText, citations)` — Task 3 hooks states into these. Errors still use the legacy inline `.status` line this task (Task 3 swaps to toasts).

- [ ] **Step 1: Rewrite `api/static/app.js`**

```javascript
const log = document.getElementById("log");
const threadList = document.getElementById("thread-list");
const providerSelect = document.getElementById("provider");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-btn");
let threadId = null; // set from the first reply; sent back to continue the thread
let inFlight = false; // guards against a second send starting a parallel thread mid-stream

// ---------- rendering helpers ----------

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollLog() { log.scrollTop = log.scrollHeight; }

function renderMarkdown(node, text) {
  if (typeof window.marked === "undefined" || typeof window.DOMPurify === "undefined") {
    node.textContent = text;
    return;
  }
  node.innerHTML = DOMPurify.sanitize(marked.parse(text));
}

function addStatus(text) {
  const node = el("div", "status", text);
  log.appendChild(node);
  scrollLog();
  return node;
}

// ---------- turn component ----------
// One .turn per exchange: user bubble, activity accordion (agent trace stays
// attached to THIS reply forever), assistant bubble, meta row.

function startTurn(userText) {
  const root = el("div", "turn");
  const user = el("div", "msg user", userText);
  const activity = document.createElement("details");
  activity.className = "activity";
  activity.hidden = true;
  const summary = el("summary", null, "working…");
  const lines = el("div", "activity-lines");
  activity.append(summary, lines);
  const assistant = el("div", "msg assistant");
  const meta = el("div", "meta-row");
  root.append(user, activity, assistant, meta);
  log.appendChild(root);
  scrollLog();
  return { root, activity, summary, lines, assistant, meta, steps: 0, raw: "", pendingText: "" };
}

function addActivityLine(turn, text) {
  turn.activity.hidden = false;
  turn.activity.open = true;
  turn.steps += 1;
  turn.lines.appendChild(el("div", "activity-line", text));
  scrollLog();
}

function finishActivity(turn) {
  if (turn.activity.hidden) return;
  turn.activity.open = false;
  turn.summary.textContent = `⚙ ${turn.steps} step${turn.steps === 1 ? "" : "s"}`;
}

function fillMeta(turn, citations, faithful) {
  for (const id of citations || []) {
    const a = el("a", "chip", id);
    a.href = `https://arxiv.org/abs/${id}`;
    a.target = "_blank";
    a.rel = "noopener";
    turn.meta.appendChild(a);
  }
  if (faithful === false) {
    turn.meta.appendChild(el("span", "chip warn", "⚠ citations unverified"));
  }
  if (navigator.clipboard && turn.raw) {
    const btn = el("button", "copy-btn", "copy");
    btn.addEventListener("click", async () => {
      await navigator.clipboard.writeText(turn.raw);
      btn.textContent = "✓ copied";
      setTimeout(() => { btn.textContent = "copy"; }, 1500);
    });
    turn.meta.appendChild(btn);
  }
}

function renderStaticTurn(userText, assistantText, citations) {
  const turn = startTurn(userText);
  turn.activity.remove(); // restored transcripts carry no activity trace
  turn.raw = assistantText;
  renderMarkdown(turn.assistant, assistantText);
  fillMeta(turn, citations || [], null); // verdicts aren't persisted — no badge
}

// ---------- providers ----------

async function loadProviders() {
  const resp = await fetch("/api/providers");
  const providers = await resp.json();
  providerSelect.replaceChildren();
  let selected = null;
  for (const p of providers) {
    const opt = document.createElement("option");
    opt.value = p.provider;
    opt.textContent = p.available ? `${p.provider} (${p.model})`
                                  : `${p.provider} — ${p.detail}`;
    opt.disabled = !p.available;
    providerSelect.appendChild(opt);
    if (p.available && (selected === null || p.is_default)) selected = p.provider;
  }
  if (selected) providerSelect.value = selected;
}

// ---------- SSE chat ----------

function parseSSE(buffer, onEvent) {
  // Returns the unconsumed tail of buffer; calls onEvent(name, data) per event.
  const events = buffer.split("\n\n");
  const tail = events.pop(); // possibly incomplete
  for (const raw of events) {
    let name = "message";
    let data = "";
    for (const line of raw.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (data) onEvent(name, JSON.parse(data));
  }
  return tail;
}

async function sendMessage() {
  if (inFlight) return;
  const message = input.value.trim();
  if (!message) return;
  input.value = "";

  const turn = startTurn(message);
  const body = { message, provider: providerSelect.value || null };
  if (threadId) body.thread_id = threadId;

  inFlight = true;
  sendBtn.disabled = true;
  input.disabled = true;

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSSE(buffer, (name, data) => {
        if (name === "status") {
          addActivityLine(turn, data.text);
        } else if (name === "delta") {
          turn.pendingText += data.text;
          turn.assistant.textContent = turn.pendingText;
          scrollLog();
        } else if (name === "turn_end") {
          if (data.has_tools) {
            // interim tool-reasoning text belongs to the trace, not the reply
            if (turn.pendingText) addActivityLine(turn, turn.pendingText);
            turn.pendingText = "";
            turn.assistant.textContent = "";
          }
        } else if (name === "done") {
          threadId = data.thread_id;
          turn.raw = data.reply;
          renderMarkdown(turn.assistant, data.reply); // authoritative full reply
          fillMeta(turn, data.citations, data.faithful);
          finishActivity(turn);
          finished = true;
          loadThreads();
        } else if (name === "error") {
          turn.assistant.textContent = "⚠ failed — try again";
          turn.assistant.classList.add("failed");
          finishActivity(turn);
          addStatus(`Chat failed: ${data.message}`);
          finished = true;
        }
      });
    }
    scrollLog();
  } catch (err) {
    turn.assistant.textContent = "⚠ failed — try again";
    turn.assistant.classList.add("failed");
    finishActivity(turn);
    addStatus(`Chat failed: ${err.message}`);
  } finally {
    inFlight = false;
    sendBtn.disabled = false;
    input.disabled = false;
  }
}

// ---------- threads ----------

async function loadThreads() {
  const resp = await fetch("/api/threads");
  const threads = await resp.json();
  threadList.replaceChildren();
  for (const t of threads) {
    const row = el("div", "thread" + (t.thread_id === threadId ? " active" : ""));
    const title = el("span", "title", t.title);
    title.title = t.title;
    row.appendChild(title);
    const del = el("button", "del", "✕");
    del.title = "Delete thread";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/threads/${t.thread_id}`, { method: "DELETE" });
      if (t.thread_id === threadId) startNewConversation();
      loadThreads();
    });
    row.appendChild(del);
    row.addEventListener("click", () => openThread(t.thread_id));
    threadList.appendChild(row);
  }
}

async function openThread(id) {
  document.body.classList.remove("sidebar-open"); // close mobile drawer
  const resp = await fetch(`/api/threads/${id}`);
  if (!resp.ok) {
    addStatus("No transcript available for this thread.");
    return;
  }
  const turns = await resp.json();
  threadId = id;
  log.replaceChildren();
  let pendingUser = null;
  for (const turn of turns) {
    if (turn.role === "user") {
      if (pendingUser !== null) renderStaticTurn(pendingUser, "", []);
      pendingUser = turn.text;
    } else {
      renderStaticTurn(pendingUser ?? "", turn.text, turn.citations || []);
      pendingUser = null;
    }
  }
  if (pendingUser !== null) renderStaticTurn(pendingUser, "", []);
  loadThreads(); // refresh active highlight
}

function startNewConversation() {
  threadId = null;
  log.replaceChildren();
  addStatus("New conversation started.");
  loadThreads();
}

// ---------- ingest ----------

async function post(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.json();
}

document.getElementById("ingest-btn").addEventListener("click", async () => {
  const query = document.getElementById("ingest-query").value.trim();
  if (!query) return;
  const status = document.getElementById("ingest-status");
  status.textContent = "Ingesting… (downloads, parses, and embeds PDFs — may take a minute)";
  try {
    const result = await post("/api/ingest", { query, max_results: 3 });
    status.textContent = `Ingested: ${result.ingested.join(", ") || "none"}` +
      (result.skipped.length ? ` | Skipped: ${result.skipped.join(", ")}` : "");
  } catch (err) {
    status.textContent = `Ingest failed: ${err.message}`;
  }
});

// ---------- wiring ----------

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
document.getElementById("new-conv-btn").addEventListener("click", startNewConversation);

loadProviders();
loadThreads();
```

- [ ] **Step 2: Verify suite**

Run: `uv run pytest -q`
Expected: all pass (no server files touched).

- [ ] **Step 3: Commit**

```bash
git add api/static/app.js
git commit -m "feat: per-reply turn component with collapsible activity accordion"
```

---

### Task 3: States — welcome card, toasts, banner, scroll pinning, input polish

**Files:**
- Modify: `api/static/app.js`
- Modify: `api/static/styles.css` (delete legacy block)

**Interfaces:**
- Consumes: Task 2's `startTurn`/`addStatus` structure; Task 1's `#toasts`, `#scroll-pill`, `#provider-banner`, `.welcome`, `.busy` CSS.
- Produces: final app.js. `addStatus` remains only for "New conversation started." — all error paths use `toast()`.

- [ ] **Step 1: Add state components to `app.js`**

1a. After `addStatus`, add:

```javascript
function toast(message) {
  const box = document.getElementById("toasts");
  const node = el("div", "toast", message);
  node.addEventListener("click", () => node.remove());
  box.appendChild(node);
  setTimeout(() => node.remove(), 6000);
}

// ---------- welcome card ----------

const EXAMPLE_QUESTIONS = [
  "What attention mechanism does the Transformer rely on?",
  "How does BERT's pre-training objective work?",
  "What problem does retrieval-augmented generation solve?",
];

function renderWelcome() {
  const card = el("div", "welcome");
  card.appendChild(el("div", "welcome-title", "Ask about the ingested papers"));
  card.appendChild(el("div", "welcome-sub",
    "Answers stream with live agent activity, [paper_id] citations, and a faithfulness check."));
  for (const q of EXAMPLE_QUESTIONS) {
    const btn = el("button", "example", q);
    btn.addEventListener("click", () => {
      input.value = q;
      sendMessage();
    });
    card.appendChild(btn);
  }
  log.appendChild(card);
}

function clearWelcome() {
  const card = log.querySelector(".welcome");
  if (card) card.remove();
}
```

1b. Scroll pinning — replace the `scrollLog` function with:

```javascript
const scrollPill = document.getElementById("scroll-pill");
let pinned = true;

function scrollLog(force = false) {
  if (force || pinned) {
    log.scrollTop = log.scrollHeight;
    scrollPill.hidden = true;
  } else {
    scrollPill.hidden = false;
  }
}

log.addEventListener("scroll", () => {
  pinned = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  if (pinned) scrollPill.hidden = true;
});
scrollPill.addEventListener("click", () => {
  pinned = true;
  scrollLog(true);
});
```

1c. Error paths → toasts. In `sendMessage`, replace both
`addStatus(`Chat failed: ${data.message}`);` and
`addStatus(`Chat failed: ${err.message}`);` with
`toast(`Chat failed: ${data.message}`);` / `toast(`Chat failed: ${err.message}`);`.
In `openThread`, replace `addStatus("No transcript available for this thread.");`
with `toast("No transcript for this thread.");`.
In the ingest handler's catch, ADD `toast(`Ingest failed: ${err.message}`);`
before the status-line assignment.

1d. Welcome integration: in `sendMessage`, after `input.value = "";` add
`clearWelcome();`. In `startNewConversation`, replace
`addStatus("New conversation started.");` with `renderWelcome();`.
At the bottom of the file, after `loadThreads();`, add `renderWelcome();`
(initial load starts a fresh conversation view).
In `openThread`, after `log.replaceChildren();` nothing extra (transcript fills it).

1e. Provider banner — replace the tail of `loadProviders` (from `if (selected) ...`) with:

```javascript
  if (selected) providerSelect.value = selected;
  const banner = document.getElementById("provider-banner");
  const none = selected === null;
  banner.hidden = !none;
  input.disabled = none;
  sendBtn.disabled = none;
```

1f. Loading states:
- Ingest: in the ingest click handler, wrap the fetch with button busy state —
  after `const status = ...` add:

```javascript
  const btn = document.getElementById("ingest-btn");
  btn.disabled = true;
  btn.classList.add("busy");
```

  and in a `finally` block (convert the try/catch to try/catch/finally):

```javascript
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
  }
```

- Thread restore: in `openThread`, before the fetch add
  `log.replaceChildren(el("div", "status", "loading…"));` (the later
  `log.replaceChildren()` clears it; on 404 restore the previous view is
  already replaced — after the toast add `startNewConversation();`).

1g. Textarea autosize — add to the wiring section:

```javascript
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 148) + "px";
});
```

(Enter/Shift+Enter handling already correct from Task 2.)

- [ ] **Step 2: Delete the legacy CSS block**

In `api/static/styles.css`, delete everything from the comment
`/* ---------- LEGACY classes for pre-Task-2 app.js — DELETE in Task 3 ---------- */`
to the end of the file, EXCEPT keep the `.status` rule (still used by
"New conversation started"-era placeholder + ingest status) — `.status` already
exists above the legacy block, so the whole legacy block goes.

- [ ] **Step 3: Verify suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add api/static/app.js api/static/styles.css
git commit -m "feat: welcome card, error toasts, provider banner, scroll pinning, textarea input"
```

---

### Task 4: README + smoke checklist

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: shipped UI behavior from Tasks 1–3.

- [ ] **Step 1: Update `README.md`**

1a. Status line → `## Status: complete (all 8 phases shipped)`

1b. Status table — add after the phase-7 row:

```markdown
| 8 | UI/UX refresh: token design system with auto/manual dark mode, per-reply agent-activity accordion, welcome/empty states, error toasts, scroll pinning, mobile drawer layout |
```

1c. Replace the "The web UI (phase 4):" block's bullet list with:

```markdown
The web UI (phases 4 + 8):
- **Per-request provider toggle** — header dropdown switches reasoning between
  Anthropic / OpenAI / local Ollama per message; a banner + disabled input
  appear when no provider is available.
- **Streaming with an activity accordion** — replies stream over SSE; each
  reply's agent trace ("calling rag_query…", "grading 8 chunks…") lives in a
  collapsible block attached to that reply, folding to "⚙ N steps" when done.
- **Citations + faithfulness** — [paper_id] chips link to arXiv; unverified
  answers get a warning chip; a copy button grabs the reply's markdown.
- **Dark mode** — follows the OS by default; header toggle cycles
  auto → light → dark (persisted).
- **Thread sidebar** — persistent conversations (list / restore / delete);
  collapses to a drawer on small screens; ingest lives at the sidebar bottom.
- Vanilla JS + vendored `marked`/`DOMPurify` — no build step, works offline.
```

- [ ] **Step 2: Full suite, then commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add README.md
git commit -m "docs: phase 8 README — UI/UX refresh feature list"
```

---

### End-of-phase browser smoke checklist (controller/human, services required)

Run with Qdrant + Ollama up, `uv run uvicorn api.main:app --reload`:

1. Stream a question end-to-end — activity lines appear inside the accordion, collapse to "⚙ N steps", re-expand on click.
2. Reply renders markdown; citation chips link to arXiv; copy button copies.
3. Theme toggle cycles auto/light/dark; choice survives reload; no flash on load.
4. 375px viewport: hamburger opens drawer; selecting a thread closes it; backdrop click closes it.
5. New conversation shows the welcome card; clicking an example sends it.
6. Stop Qdrant mid-session → next question yields an error toast + failed turn, input re-enabled.
7. Restore an old thread — turns render with citations, no activity blocks; unknown thread → toast.
8. Scroll up during a stream → "↓ new messages" pill instead of yank; click resumes pinning.
9. Kill Ollama with no keys configured → reload shows the amber provider banner, input disabled.
