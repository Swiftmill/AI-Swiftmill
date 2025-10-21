const API_BASE = "http://localhost:8000";
const HISTORY_KEY = "ai-local-history";
const USER_KEY = "ai-local-user-id";

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const webToggle = document.getElementById("web-toggle");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const spinner = document.getElementById("spinner");
const template = document.getElementById("message-template");

const userId = getOrCreateUserId();
let history = loadHistory();
renderHistory();

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  appendMessage("toi", message);
  history.push({ role: "user", text: message, time: Date.now() });
  persistHistory();
  messageInput.value = "";

  toggleSpinner(true);
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        user: userId,
        allow_web: webToggle.checked,
      }),
    });

    if (!response.ok) {
      throw new Error(`Erreur ${response.status}`);
    }

    const payload = await response.json();
    const fullAnswer = buildAnswerWithSources(payload);

    appendMessage("assistant", fullAnswer);
    history.push({ role: "assistant", text: fullAnswer, time: Date.now() });
    persistHistory();
  } catch (error) {
    appendMessage("assistant", `❌ ${error.message}`);
  } finally {
    toggleSpinner(false);
  }
});

uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleFiles);

["dragenter", "dragover"].forEach((eventName) => {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.hidden = false;
    dropZone.classList.add("drop-zone--active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("drop-zone--active");
    dropZone.hidden = eventName === "dragleave";
  });
});

dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer?.files;
  if (files?.length) {
    uploadFiles(files);
  }
  dropZone.hidden = true;
});

function buildAnswerWithSources(payload) {
  if (!payload.sources?.length) {
    return payload.answer;
  }
  const sourcesList = payload.sources
    .map((source) => `- ${source.title} (${source.url})`)
    .join("\n");
  return `${payload.answer}\n\nSources:\n${sourcesList}`;
}

async function handleFiles(event) {
  if (event.target.files?.length) {
    await uploadFiles(event.target.files);
    fileInput.value = "";
  }
}

async function uploadFiles(fileList) {
  const form = new FormData();
  Array.from(fileList).forEach((file) => form.append("files", file));

  toggleSpinner(true);
  appendMessage("assistant", "📁 Téléversement en cours...");
  try {
    const response = await fetch(`${API_BASE}/ingest`, {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      throw new Error(`Ingestion impossible (${response.status})`);
    }

    const payload = await response.json();
    const files = (payload.files || []).join(", ");
    appendMessage("assistant", `✅ Documents ajoutés: ${files}`);
    history.push({ role: "assistant", text: `✅ Documents ajoutés: ${files}`, time: Date.now() });
    persistHistory();
  } catch (error) {
    appendMessage("assistant", `❌ ${error.message}`);
  } finally {
    toggleSpinner(false);
  }
}

function appendMessage(role, text, timestamp = Date.now()) {
  const clone = template.content.firstElementChild.cloneNode(true);
  clone.querySelector(".message__role").textContent = role;
  clone.querySelector(".message__time").textContent = new Date(timestamp).toLocaleTimeString();
  clone.querySelector(".message__content").textContent = text;
  chatLog.appendChild(clone);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderHistory() {
  chatLog.innerHTML = "";
  history.forEach((item) =>
    appendMessage(item.role === "user" ? "toi" : "assistant", item.text, item.time)
  );
}

function persistHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-50)));
}

function loadHistory() {
  const raw = localStorage.getItem(HISTORY_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch (error) {
    console.warn("History parse error", error);
    return [];
  }
}

function getOrCreateUserId() {
  let id = localStorage.getItem(USER_KEY);
  if (id) return id;
  id = crypto.randomUUID();
  localStorage.setItem(USER_KEY, id);
  return id;
}

function toggleSpinner(visible) {
  spinner.classList.toggle("hidden", !visible);
}
