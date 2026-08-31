function handleParse() {
  const rawText = document.getElementById("raw-haplotype").value;
  if (!rawText.trim()) {
    updateTreeStatus("Поле ввода пусто! Вставьте сырые данные.", "status-error");
    return;
  }
  RAW_CSV_COLS.forEach(m => {
    const el = document.getElementById(`marker-${m}`);
    if (el) el.value = "";
  });
  const values = parseHaplotypeString(rawText);
  const modifiedInputs = [];
  const totalToFill = Math.min(values.length, RAW_CSV_COLS.length);
  for (let i = 0; i < totalToFill; i++) {
    const inputField = document.getElementById(`marker-${RAW_CSV_COLS[i]}`);
    if (inputField) {
      inputField.value = values[i];
      inputField.classList.add("highlight-pulse");
      modifiedInputs.push(inputField);
    }
  }
  setTimeout(() => {
    modifiedInputs.forEach(el => el.classList.remove("highlight-pulse"));
  }, 1000);
  updateTreeStatus(`✓ Успешно обработано. Заполнено полей: ${modifiedInputs.length}`, "status-success");
}

async function handlePredict() {
  const confidence = document.getElementById("confidence-level").value;
  const treeResultContainer = document.getElementById("tree-result");
  const haplotypeData = {};
  RAW_CSV_COLS.forEach(marker => {
    const val = document.getElementById(`marker-${marker}`).value.trim();
    if (val === "") {
      haplotypeData[marker] = null;
    } else if (val.includes('-')) {
      haplotypeData[marker] = val;
    } else {
      const parsed = parseInt(val, 10);
      haplotypeData[marker] = isNaN(parsed) ? val : parsed;
    }
  });
  const payload = {
    confidence: parseFloat(confidence),
    haplotype: haplotypeData
  };
  updateTreeStatus("Выполняется запрос к серверу прогнозирования...", "status-loading");
  try {
    const resultData = await fetchPrediction(payload);
    treeResultContainer.innerHTML = "";
    treeResultContainer.appendChild(buildTreeHTML(resultData));
  } catch (error) {
    console.error(error);
    updateTreeStatus(`Ошибка при расчете: ${error.message}`, "status-error");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  renderMarkerFields();
  document.getElementById("parse-btn").addEventListener("click", handleParse);
  document.getElementById("predict-btn").addEventListener("click", handlePredict);
});
