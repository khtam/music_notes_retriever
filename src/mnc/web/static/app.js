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
  $("#result-meta").textContent = meta.join("  ·  ");

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
