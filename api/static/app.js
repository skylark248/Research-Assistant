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

// ---------- ingest (unchanged behavior) ----------

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
