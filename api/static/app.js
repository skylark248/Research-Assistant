const log = document.getElementById("log");
const threadList = document.getElementById("thread-list");
const providerSelect = document.getElementById("provider");
let threadId = null; // set from the first reply; sent back to continue the thread

// ---------- rendering ----------

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

function addUser(text) {
  const node = el("div", "user", `You: ${text}`);
  log.appendChild(node);
  scrollLog();
}

function addBotMarkdown(text, citations) {
  const node = el("div", "bot");
  renderMarkdown(node, text);
  log.appendChild(node);
  addCitations(citations);
  scrollLog();
}

function addCitations(citations) {
  if (!citations || !citations.length) return;
  const row = el("div", "citations");
  for (const id of citations) {
    const a = el("a", null, id);
    a.href = `https://arxiv.org/abs/${id}`;
    a.target = "_blank";
    a.rel = "noopener";
    row.appendChild(a);
  }
  log.appendChild(row);
}

function addStatus(text) {
  const node = el("div", "status", text);
  log.appendChild(node);
  scrollLog();
  return node;
}

function addActivity(text) {
  const node = el("div", "activity", text);
  log.appendChild(node);
  scrollLog();
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

let inFlight = false; // guards against a second send starting a parallel thread mid-stream

async function sendMessage() {
  if (inFlight) return;

  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addUser(message);

  const pending = el("div", "bot"); // live-updating bubble for streamed deltas
  log.appendChild(pending);
  let pendingText = "";
  const thinking = addStatus("thinking…");

  const body = { message, provider: providerSelect.value || null };
  if (threadId) body.thread_id = threadId;

  inFlight = true;
  const sendBtn = document.getElementById("chat-btn");
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
    thinking.remove();
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
          addActivity(data.text);
        } else if (name === "delta") {
          pendingText += data.text;
          // keep the streaming bubble below any activity lines added since
          if (pending !== log.lastChild) log.appendChild(pending);
          pending.textContent = pendingText;
          scrollLog();
        } else if (name === "turn_end") {
          if (data.has_tools) {
            if (pendingText) addActivity(pendingText); // tool-reasoning text → activity feed
            pendingText = "";
            pending.textContent = "";
          }
          // final turn (has_tools=false): leave streamed text in place;
          // `done` re-renders it as markdown in the same bubble — no flicker
        } else if (name === "done") {
          threadId = data.thread_id;
          if (pending !== log.lastChild) log.appendChild(pending); // e.g. no deltas streamed
          renderMarkdown(pending, data.reply); // authoritative full reply
          addCitations(data.citations);
          finished = true;
          loadThreads();
        } else if (name === "error") {
          pending.remove();
          addStatus(`Chat failed: ${data.message}`);
          finished = true;
        }
      });
    }
    scrollLog();
  } catch (err) {
    thinking.remove();
    pending.remove();
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
  const resp = await fetch(`/api/threads/${id}`);
  if (!resp.ok) {
    addStatus("No transcript available for this thread.");
    return;
  }
  const turns = await resp.json();
  threadId = id;
  log.replaceChildren();
  for (const turn of turns) {
    if (turn.role === "user") addUser(turn.text);
    else addBotMarkdown(turn.text, turn.citations || []);
  }
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

document.getElementById("chat-btn").addEventListener("click", sendMessage);
document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
document.getElementById("new-conv-btn").addEventListener("click", startNewConversation);

loadProviders();
loadThreads();
