import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

// Live metadata preview for "Load Image with Metadata (Image Saver)": shows the selected file's generation
// metadata without running the workflow, as a table or as the raw A1111 text, both copyable.
// File content is untrusted: only ever inserted via textContent/value, never as HTML.

// Nodes that get the preview, mapped to the name of the widget holding the selected file.
// Other loaders (e.g. for video, in forks) can reuse the preview by adding an entry here.
const PREVIEW_NODES = {
    "Load Image with Metadata (Image Saver)": "image",
};
const POLL_MS = 400; // also catches upload, paste, mask editor and workflow loading, which don't all fire callbacks
const MIN_HEIGHT = 30; // just the button bar
const MAX_HEIGHT = 480; // longer tables scroll
const TEXT_VIEW_HEIGHT = 200;

const STYLE = `
.isv-meta { display: flex; flex-direction: column; gap: 4px; height: 100%; box-sizing: border-box; font-size: 11px;
  color: var(--input-text, #ddd); }
.isv-meta-bar { display: flex; gap: 4px; align-items: center; flex: none; }
.isv-meta-bar button { font-size: 11px; padding: 1px 6px; cursor: pointer; border-radius: 4px;
  border: 1px solid var(--border-color, #555); background: var(--comfy-input-bg, #222); color: inherit; }
.isv-meta-bar button.active { background: var(--border-color, #555); }
.isv-meta-status { flex: 1; text-align: right; opacity: 0.7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.isv-meta-warning { flex: none; color: var(--error-text, #f0b060); white-space: pre-wrap; }
.isv-meta-body { flex: 1; min-height: 0; overflow: auto; border-radius: 4px; background: var(--comfy-input-bg, #222); }
.isv-meta-body table { border-collapse: collapse; width: 100%; }
.isv-meta-body td { vertical-align: top; padding: 2px 4px; border-bottom: 1px solid var(--border-color, #333); }
.isv-meta-body td.k { white-space: nowrap; opacity: 0.7; }
.isv-meta-body td.v { white-space: pre-wrap; word-break: break-word; user-select: text; cursor: text; width: 100%; }
.isv-meta-body td.c button { font-size: 10px; padding: 0 4px; cursor: pointer; border: none; background: none;
  color: inherit; opacity: 0.6; }
.isv-meta-body td.c button:hover { opacity: 1; }
.isv-meta-body textarea { width: 100%; height: 100%; box-sizing: border-box; resize: none; border: none; outline: none;
  background: transparent; color: inherit; font: 11px monospace; padding: 4px; }
`;

function addStyle() {
    if (document.getElementById("isv-meta-style")) return;
    const style = document.createElement("style");
    style.id = "isv-meta-style";
    style.textContent = STYLE;
    document.head.appendChild(style);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        // Clipboard API needs a secure context (https or localhost); fall back for LAN access
        const area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        const ok = document.execCommand("copy");
        area.remove();
        return ok;
    }
}

function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

function createPreview(node, fileWidgetName) {
    const root = el("div", "isv-meta");
    const bar = el("div", "isv-meta-bar");
    const tableButton = el("button", "", "Table");
    const textButton = el("button", "", "Text");
    const copyButton = el("button", "", "Copy");
    copyButton.title = "Copy the A1111 parameters (or the table, if the file has none)";
    const status = el("span", "isv-meta-status");
    bar.append(tableButton, textButton, copyButton, status);
    const warning = el("div", "isv-meta-warning");
    warning.hidden = true;
    const body = el("div", "isv-meta-body");
    root.append(bar, warning, body);

    // Let text selection and scrolling work instead of dragging/zooming the canvas
    for (const type of ["pointerdown", "mousedown", "wheel"]) {
        body.addEventListener(type, (event) => event.stopPropagation());
    }

    let data = null;
    const view = () => node.properties?.metadataView === "text" ? "text" : "table";
    const flash = (button, label) => {
        const original = button.textContent;
        button.textContent = label;
        setTimeout(() => { button.textContent = original; }, 900);
    };
    const tableText = () => (data?.rows ?? []).map(([key, value]) => `${key}: ${value}`).join("\n");

    function render() {
        tableButton.classList.toggle("active", view() === "table");
        textButton.classList.toggle("active", view() === "text");
        warning.hidden = !data?.warning;
        warning.textContent = data?.warning ?? "";
        body.replaceChildren();
        if (!data) return;
        if (view() === "text") {
            const area = el("textarea");
            area.readOnly = true;
            area.value = data.parameters || tableText();
            body.append(area);
            return;
        }
        const table = el("table");
        for (const [key, value] of data.rows ?? []) {
            const row = el("tr");
            const copy = el("button", "", "copy");
            copy.title = `Copy ${key}`;
            copy.addEventListener("click", async () => flash(copy, await copyText(value) ? "✓" : "✗"));
            const copyCell = el("td", "c");
            copyCell.append(copy);
            row.append(el("td", "k", key), el("td", "v", value), copyCell);
            table.append(row);
        }
        body.append(table);
    }

    const setView = (value) => {
        node.properties = node.properties ?? {};
        node.properties.metadataView = value;
        render();
    };
    tableButton.addEventListener("click", () => setView("table"));
    textButton.addEventListener("click", () => setView("text"));
    copyButton.addEventListener("click", async () => {
        const text = data?.parameters || tableText();
        if (text) flash(copyButton, await copyText(text) ? "Copied" : "Failed");
    });

    let current = null;
    let request = 0;
    async function update(filename) {
        current = filename;
        const id = ++request;
        if (!filename) {
            data = null;
            status.textContent = "";
            render();
            return;
        }
        status.textContent = "reading…";
        try {
            const response = await api.fetchApi(`/image_saver/image_metadata?filename=${encodeURIComponent(filename)}`);
            const json = await response.json();
            if (id !== request) return; // a newer selection is in flight
            data = response.ok ? json : { rows: [], parameters: "", warning: json.error ?? `HTTP ${response.status}` };
            status.textContent = data.source === "parameters" ? "A1111 parameters" : data.source === "workflow" ? "ComfyUI workflow" : "";
        } catch (error) {
            if (id !== request) return;
            data = { rows: [], parameters: "", warning: `Could not read metadata: ${error}` };
            status.textContent = "";
        }
        render();
    }

    // The widget is exactly as tall as its content (up to a cap), so the node grows to fit the table and any
    // extra height from resizing the node goes to the image preview instead of empty table space.
    let contentHeight = MIN_HEIGHT;
    function measure() {
        if (!bar.offsetHeight) return; // not laid out (e.g. hidden while zoomed out)
        const content = body.firstElementChild;
        const bodyHeight = !content ? 0 : view() === "text" ? TEXT_VIEW_HEIGHT : content.scrollHeight + 2;
        const warningHeight = warning.hidden ? 0 : warning.offsetHeight + 4;
        const margin = 2 * (previewWidget?.margin ?? 10); // the canvas insets DOM widgets by this margin
        const height = Math.max(MIN_HEIGHT, Math.min(MAX_HEIGHT, bar.offsetHeight + warningHeight + 4 + bodyHeight)) + margin;
        if (Math.abs(height - contentHeight) < 2) return;
        contentHeight = height;
        const minimum = node.computeSize();
        if (node.size[1] < minimum[1]) node.setSize([node.size[0], minimum[1]]);
        node.setDirtyCanvas?.(true, true);
    }

    let previewWidget = null;
    // Keep the table below the image: the image preview widget is added (again) whenever an image loads
    function keepLast() {
        const widgets = node.widgets;
        const index = widgets?.indexOf(previewWidget) ?? -1;
        if (index !== -1 && index !== widgets.length - 1) {
            widgets.splice(index, 1);
            widgets.push(previewWidget);
            node.setDirtyCanvas?.(true, true);
        }
    }

    const imageWidget = () => node.widgets?.find((widget) => widget.name === fileWidgetName);
    const timer = setInterval(() => {
        keepLast();
        measure(); // also catches re-wrapping when the node gets narrower or wider
        const value = imageWidget()?.value ?? null;
        if (value !== current) update(value);
    }, POLL_MS);

    return { root, stop: () => clearInterval(timer), attach: (widget) => { previewWidget = widget; }, height: () => contentHeight };
}

app.registerExtension({
    name: "ComfyUI-Image-Saver.LoadImageWithMetadata",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const fileWidgetName = PREVIEW_NODES[nodeData.name];
        if (!fileWidgetName) return;
        addStyle();

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const preview = createPreview(this, fileWidgetName);
            const widget = this.addDOMWidget("metadata_preview", "image_saver_metadata", preview.root, {
                serialize: false,
                getMinHeight: () => preview.height(),
                getMaxHeight: () => preview.height(),
            });
            widget.serialize = false; // display only, not a node input
            preview.attach(widget);

            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                preview.stop();
                return onRemoved?.apply(this, arguments);
            };
            return result;
        };
    },
});
