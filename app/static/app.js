const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, preventDefaults, false);
});

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, () => {
    dropZone.classList.add("highlight");
  }, false);
});

["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, () => {
    dropZone.classList.remove("highlight");
  }, false);
});

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("drop", handleDrop, false);
fileInput.addEventListener("change", handleFileSelect, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files && files[0]) {
    uploadFile(files[0]);
  }
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files && files[0]) {
    uploadFile(files[0]);
  }
}

function setStatus(text) {
  statusEl.textContent = text;
}

function clearResults() {
  resultsEl.innerHTML = "";
}

function renderResults(data) {
  clearResults();

  const heading = document.createElement("h2");
  heading.textContent = `Results for: ${data.file_name}`;
  resultsEl.appendChild(heading);

  const models = data.models || {};

  Object.keys(models).forEach(modelName => {
    const modelSection = document.createElement("div");
    modelSection.className = "model-section";

    const title = document.createElement("h3");
    title.textContent = `Model: ${modelName}`;
    modelSection.appendChild(title);

    const pages = models[modelName];
    pages.forEach(page => {
      const pageDiv = document.createElement("div");
      pageDiv.className = "page-result";

      const pageTitle = document.createElement("h4");
      pageTitle.textContent = `Page ${page.page}`;
      pageDiv.appendChild(pageTitle);

      const pre = document.createElement("pre");
      pre.textContent = page.text || "";
      pageDiv.appendChild(pre);

      modelSection.appendChild(pageDiv);
    });

    resultsEl.appendChild(modelSection);
  });
}

async function uploadFile(file) {
  clearResults();
  setStatus(`Uploading "${file.name}" and running OCR...`);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/ocr", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Server error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    setStatus("OCR complete.");
    renderResults(data);
  } catch (err) {
    console.error(err);
    setStatus("Error: " + err.message);
  }
}
