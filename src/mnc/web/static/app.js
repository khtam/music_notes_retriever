"use strict";

const $ = (sel) => document.querySelector(sel);

const cards = {
  input: $("#input-card"),
  progress: $("#progress-card"),
  error: $("#error-card"),
  result: $("#result"),
};

function show(name) {
  for (const [key, el] of Object.entries(cards)) {
    el.classList.toggle("hidden", key !== name);
  }
}

// --- tabs -------------------------------------------------------------
let activeTab = "youtube";
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) =>
      p.classList.toggle("hidden", p.dataset.panel !== activeTab)
    );
  });
});

// --- LLM options ------------------------------------------------------
const llmCheck = $("#llm-check");
const GROUP_LABELS = { major: "Major players", regional: "Low-cost & regional", local: "Local & custom" };
let providers = [];

function syncLlmFields() {
  const off = !llmCheck.checked;
  $("#llm-fields").classList.toggle("disabled", off);
  document.querySelectorAll("#llm-fields select, #llm-fields input").forEach((el) => {
    el.disabled = off;
  });
}
llmCheck.addEventListener("change", syncLlmFields);
syncLlmFields();

function syncProviderFields() {
  const spec = providers.find((p) => p.id === $("#llm-provider").value);
  const keyInput = $("#llm-api-key");
  const docsLink = $("#llm-docs-link");
  const baseUrlRow = $("#llm-base-url-row");
  const modelInput = $("#llm-model");

  keyInput.placeholder = spec && spec.key_prefix ? `${spec.key_prefix}  (blank = server’s key)` : "blank = server’s key";
  keyInput.disabled = !llmCheck.checked || (spec ? spec.needs_key === false : false);

  if (spec && spec.docs_url) {
    docsLink.href = spec.docs_url;
    docsLink.classList.remove("hidden");
  } else {
    docsLink.classList.add("hidden");
  }

  baseUrlRow.classList.toggle("hidden", !(spec && spec.editable_base_url));
  $("#llm-base-url").value = (spec && spec.base_url) || "";

  modelInput.placeholder = spec && spec.default_model ? spec.default_model : "(provider default)";
}

async function loadProviders() {
  try {
    const res = await fetch("/api/providers");
    providers = await res.json();
  } catch (err) {
    return; // provider list is a UX nicety; the form still works without it
  }
  const select = $("#llm-provider");
  for (const group of ["major", "regional", "local"]) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = GROUP_LABELS[group];
    for (const spec of providers.filter((p) => p.group === group)) {
      const option = document.createElement("option");
      option.value = spec.id;
      option.textContent = spec.label;
      optgroup.appendChild(option);
    }
    select.appendChild(optgroup);
  }
  syncProviderFields();
}
$("#llm-provider").addEventListener("change", syncProviderFields);
loadProviders();

// --- job submission ---------------------------------------------------
let pollTimer = null;

$("#job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = new FormData();

  if (activeTab === "youtube") {
    const url = form.url.value.trim();
    if (!url) return alert("Paste a YouTube URL first.");
    data.append("url", url);
  } else {
    const file = form.file.files[0];
    if (!file) return alert("Choose an audio or video file first.");
    data.append("file", file);
  }
  for (const name of ["title", "split_point", "tempo", "min_note_length", "onset_threshold"]) {
    const value = form[name].value;
    if (value) data.append(name, value);
  }
  const lyricsText = form.lyrics_text.value;
  if (lyricsText.trim()) data.append("lyrics_text", lyricsText);
  for (const name of ["lyrics", "structure", "dedup", "llm"]) {
    data.append(name, form[name].checked ? "true" : "false");
  }
  if (form.llm.checked) {
    if (form.llm_provider.value) data.append("llm_provider", form.llm_provider.value);
    const key = form.llm_api_key.value.trim();
    if (key) data.append("llm_api_key", key);
    const model = form.llm_model.value.trim();
    if (model) data.append("llm_model", model);
    const baseUrl = form.llm_base_url.value.trim();
    if (baseUrl) data.append("llm_base_url", baseUrl);
  }

  show("progress");
  $("#progress-stage").textContent = "Uploading…";
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: data });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server error (${res.status})`);
    }
    const job = await res.json();
    poll(job.id);
  } catch (err) {
    fail(err.message);
  }
});

function poll(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error(`Lost the job (${res.status})`);
      const job = await res.json();
      $("#progress-stage").textContent = job.stage + "…";
      if (job.status === "done") {
        clearInterval(pollTimer);
        renderResult(job);
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        fail(job.error || "Transcription failed.");
      }
    } catch (err) {
      clearInterval(pollTimer);
      fail(err.message);
    }
  }, 1500);
}

function fail(message) {
  $("#error-message").textContent = message;
  show("error");
}

$("#error-retry").addEventListener("click", () => show("input"));
$("#new-job").addEventListener("click", () => {
  $("#sheet-container").innerHTML = "";
  show("input");
});

// --- rendering --------------------------------------------------------
async function renderResult(job) {
  $("#result-title").textContent = job.title || "Transcription";
  const meta = [];
  if (job.tempo_bpm) meta.push(`♩ = ${Math.round(job.tempo_bpm)}`);
  if (job.key_name) meta.push(job.key_name);
  if (job.n_notes) meta.push(`${job.n_notes} notes`);
  if (job.duration_seconds) meta.push(`${Math.round(job.duration_seconds)}s of audio`);
  if (job.n_lyric_words) {
    const details = [job.lyrics_language, job.lyrics_source].filter(Boolean).join(", ");
    meta.push(`${job.n_lyric_words} lyric words` + (details ? ` (${details})` : ""));
  }
  if (job.n_chord_symbols) meta.push(`${job.n_chord_symbols} chord symbols`);
  $("#result-meta").textContent = meta.join("  ·  ");

  const warningEl = $("#result-warning");
  if (job.lyrics_source === "mapped to melody notes") {
    warningEl.textContent =
      "Lyric timing is approximate: vocal alignment failed, so words were placed one per melody note.";
    warningEl.classList.remove("hidden");
  } else {
    warningEl.classList.add("hidden");
    warningEl.textContent = "";
  }

  const structureEl = $("#result-structure");
  if (job.sections && job.sections.length) {
    // Show which engine labeled the sections ("llm:anthropic" / "heuristic"),
    // so a pasted API key visibly took effect.
    const method = job.structure_method ? ` (${job.structure_method})` : "";
    structureEl.textContent = `Structure${method}: ` + job.sections.join("  ·  ");
    structureEl.classList.remove("hidden");
  } else {
    structureEl.classList.add("hidden");
    structureEl.textContent = "";
  }

  $("#download-xml").href = `/api/jobs/${job.id}/musicxml`;
  $("#download-midi").href = `/api/jobs/${job.id}/midi`;

  show("result");

  const container = $("#sheet-container");
  container.innerHTML = "";
  const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
    autoResize: true,
    drawTitle: true,
    drawSubtitle: false,
    drawComposer: false,
    drawingParameters: "default",
  });
  try {
    const xml = await (await fetch(`/api/jobs/${job.id}/musicxml`)).text();
    await osmd.load(xml);
    osmd.render();
  } catch (err) {
    container.innerHTML =
      `<p class="hint">Could not render the score in the browser (${err.message}). ` +
      `The MusicXML download above will still open in MuseScore.</p>`;
  }
}
