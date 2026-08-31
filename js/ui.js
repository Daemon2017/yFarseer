function renderMarkerFields() {
  const container = document.getElementById("markers-container");
  if (!container) return;
  container.innerHTML = "";
  RAW_CSV_COLS.forEach((marker, index) => {
    if (PANEL_BOUNDARIES[index]) {
      createPanelDivider(container, PANEL_BOUNDARIES[index]);
    }
    const fieldWrapper = document.createElement("div");
    fieldWrapper.className = "marker-field";
    const label = document.createElement("label");
    label.innerText = marker;
    label.setAttribute("for", `marker-${marker}`);
    const input = document.createElement("input");
    input.type = "text";
    input.id = `marker-${marker}`;
    input.placeholder = "-";
    fieldWrapper.appendChild(label);
    fieldWrapper.appendChild(input);
    container.appendChild(fieldWrapper);
  });
}

function createPanelDivider(container, titleText) {
  const divider = document.createElement("div");
  divider.className = "panel-divider";
  divider.innerText = titleText;
  container.appendChild(divider);
}

function buildTreeHTML(node) {
  const rootUl = document.createElement("ul");
  rootUl.className = "tree-flat-chain";
  const nodesArray = [];
  function traverse(currentNode) {
    if (!currentNode) return;
    nodesArray.push(currentNode);
    if (currentNode.children && currentNode.children.length > 0) {
      traverse(currentNode.children[0]);
    }
  }
  traverse(node);
  nodesArray.forEach((currentNode, index) => {
    const li = document.createElement("li");
    const nodeBox = document.createElement("div");
    nodeBox.className = "node-box";
    const nameSpan = document.createElement("span");
    nameSpan.className = "node-name";
    nameSpan.innerText = currentNode.name || "Unknown";
    const probSpan = document.createElement("span");
    probSpan.className = "node-prob";
    const scoreVal = currentNode.score !== undefined
      ? (currentNode.score <= 1 ? (currentNode.score * 100).toFixed(1) + "%" : currentNode.score)
      : "N/A";
    probSpan.innerText = `P: ${scoreVal}`;
    nodeBox.appendChild(nameSpan);
    nodeBox.appendChild(probSpan);
    li.appendChild(nodeBox);
    rootUl.appendChild(li);
    if (index < nodesArray.length - 1) {
      const arrowLi = document.createElement("li");
      arrowLi.className = "tree-arrow";
      arrowLi.innerText = "➔";
      rootUl.appendChild(arrowLi);
    }
  });
  return rootUl;
}

function updateTreeStatus(message, className = "") {
  const container = document.getElementById("tree-result");
  if (!container) return;
  container.innerHTML = `<p class="${className}">${message}</p>`;
}
