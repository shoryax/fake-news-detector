// Utility: select helpers
const $ = (sel, ctx = document) => ctx.querySelector(sel)

const textarea = $("#newsText")
const charCount = $("#charCount")
const form = $("#analyzeForm")
const submitBtn = $("#submitBtn")
const loader = $("#loader")
const result = $("#result")
const sampleBtn = $("#sampleBtn")
const clearBtn = $("#clearBtn")

const SAMPLE =
  "BREAKING: Scientists confirm water has been discovered on the Sun's surface in unprecedented solar study."

// Live character count
const updateCount = () => {
  const len = textarea.value.length
  charCount.textContent = len.toString()
}
textarea.addEventListener("input", updateCount)

// Keyboard shortcut: Ctrl/Cmd + Enter to submit
textarea.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault()
    form.requestSubmit()
  }
})

sampleBtn.addEventListener("click", () => {
  textarea.value = SAMPLE
  updateCount()
  textarea.focus()
})
clearBtn.addEventListener("click", () => {
  textarea.value = ""
  updateCount()
  textarea.focus()
  renderPlaceholder()
})

function renderPlaceholder() {
  result.innerHTML = `<div class="placeholder">Submit text to see results here.</div>`
}

let progressTimer = null
function startIndeterminateProgress() {
  const bar = $(".bar")
  if (!bar) return
  let width = 8
  clearInterval(progressTimer)
  progressTimer = setInterval(() => {
    width += Math.random() * 7
    if (width > 80) width = 80
    bar.style.width = `${width}%`
  }, 200)
}
function stopIndeterminateProgress() {
  clearInterval(progressTimer)
  progressTimer = null
}

function renderResult({ prediction, confidence, processed_text }) {
  const isReal = (prediction || "").toLowerCase() === "real"
  const isFake = (prediction || "").toLowerCase() === "fake"
  const badgeClass = isReal ? "real" : isFake ? "fake" : ""
  const progressClass = isReal ? "real" : isFake ? "fake" : ""

  const pct = Math.max(0, Math.min(100, Math.round((confidence || 0) * 100)))

  result.innerHTML = `
    <article class="card" aria-live="polite">
      <header class="card-head">
        <div class="badge ${badgeClass}" aria-label="Prediction">
          <span class="dot" aria-hidden="true"></span>
          <span>${prediction ? prediction.toUpperCase() : "UNKNOWN"} NEWS</span>
        </div>
      </header>
      <div class="card-body">
        <div class="row">
          <label>Confidence</label>
          <div class="progress ${progressClass}" aria-label="Confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
            <div class="bar" style="width: ${pct}%"></div>
          </div>
          <div class="kv">
            <div class="key">Score</div>
            <div class="val">${pct}%</div>
            <div class="key">Class</div>
            <div class="val">${prediction || "N/A"}</div>
          </div>
        </div>
        <div class="row">
          <label>Processed preview</label>
          <div class="kv">
            <div class="key">Excerpt</div>
            <div class="val">${processed_text || "—"}</div>
          </div>
        </div>
      </div>
    </article>
  `
}

function renderError(message) {
  result.innerHTML = `<div class="error" role="alert">Error: ${message}</div>`
}

function renderAnalyzing() {
  result.innerHTML = `
    <article class="card" aria-busy="true" aria-live="polite">
      <header class="card-head">
        <div class="badge" style="color: var(--primary); background: rgba(37,99,235,0.1); border: 1px solid rgba(37,99,235,0.25);">
          <span class="dot" style="background: var(--primary)"></span>
          <span>Analyzing</span>
        </div>
      </header>
      <div class="card-body">
        <div class="row">
          <label>Confidence</label>
          <div class="progress" aria-hidden="true">
            <div class="bar" style="width: 0%"></div>
          </div>
          <div class="kv">
            <div class="key">Score</div>
            <div class="val muted">…</div>
            <div class="key">Class</div>
            <div class="val muted">…</div>
          </div>
        </div>
        <div class="row">
          <label>Processed preview</label>
          <div class="kv">
            <div class="key">Excerpt</div>
            <div class="val muted">Extracting features…</div>
          </div>
        </div>
      </div>
    </article>
  `
  // Start animated fill
  startIndeterminateProgress()
}

// Submit handler
form.addEventListener("submit", async (e) => {
  e.preventDefault()
  const text = textarea.value.trim()
  if (!text) {
    renderError("Please enter some text to analyze.")
    textarea.focus()
    return
  }

  submitBtn.disabled = true
  loader.classList.add("active")
  renderAnalyzing()

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })

    const data = await res.json().catch(() => ({}))

    stopIndeterminateProgress()

    if (!res.ok || data.error) {
      const msg = data?.error || `Request failed with status ${res.status}`
      renderError(msg)
    } else {
      renderResult({
        prediction: data.prediction,
        confidence: Number(data.confidence) || 0,
        processed_text: data.processed_text,
      })
    }
  } catch (err) {
    stopIndeterminateProgress()
    renderError(err?.message || "Network error")
  } finally {
    loader.classList.remove("active")
    submitBtn.disabled = false
  }
})

// Initialize
renderPlaceholder()
updateCount()
