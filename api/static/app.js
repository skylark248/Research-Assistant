const log = document.getElementById("log");
let threadId = null; // set from the first reply; sent back to continue the thread

function append(cls, text) {
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

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

document.getElementById("chat-btn").addEventListener("click", async () => {
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  append("user", `You: ${message}`);
  append("status", "thinking…");
  try {
    const body = threadId ? { message, thread_id: threadId } : { message };
    const result = await post("/api/chat", body);
    threadId = result.thread_id;
    log.lastChild.remove();
    append("bot", `Assistant: ${result.reply}`);
  } catch (err) {
    log.lastChild.remove();
    append("status", `Chat failed: ${err.message}`);
  }
});

document.getElementById("new-conv-btn").addEventListener("click", () => {
  threadId = null;
  log.replaceChildren();
  append("status", "New conversation started.");
});
