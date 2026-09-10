// Receives worker.py's evaluate_js(...) pushes. This is the frontend half
// of the WebTQDM/_stage_fn bridge -- the evaluate_js analog of the wx GUI's
// EVT_TQDM/EVT_IW3_STAGE handlers (on_tqdm/on_stage_change in gui.py).
window.IW3Progress = (function () {
  var total = 0;
  var current = 0;
  var onDone = null;
  var onUpdateLog = null;
  var onUpdateDone = null;
  var onToolLog = null;
  var onToolDone = null;

  function setOnDone(fn) { onDone = fn; }
  function setOnUpdateLog(fn) { onUpdateLog = fn; }
  function setOnUpdateDone(fn) { onUpdateDone = fn; }
  function setOnToolLog(fn) { onToolLog = fn; }
  function setOnToolDone(fn) { onToolDone = fn; }

  function fillEl() { return document.getElementById("progressFill"); }
  function pctEl() { return document.getElementById("progressPct"); }
  function statusEl() { return document.getElementById("statusText"); }

  function render() {
    var pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
    fillEl().style.width = pct + "%";
    pctEl().textContent = total > 0 ? (pct + "%") : "";
  }

  function handle(channel, payload) {
    if (channel === "progress") {
      if (payload.type === "init") {
        total = payload.total || 0;
        current = 0;
        statusEl().textContent = payload.desc || "Working...";
      } else if (payload.type === "update") {
        current += payload.n || 1;
      } else if (payload.type === "close") {
        current = total;
      }
      render();
    } else if (channel === "stage") {
      statusEl().textContent = payload.name;
    } else if (channel === "done") {
      statusEl().textContent = payload.ok ? "Done" : ("Error: " + payload.error);
      if (onDone) onDone(payload);
    } else if (channel === "update_log") {
      if (onUpdateLog) onUpdateLog(payload.line);
    } else if (channel === "update_done") {
      if (onUpdateDone) onUpdateDone(payload);
    } else if (channel === "tool_log") {
      if (onToolLog) onToolLog(payload.tool, payload.line);
    } else if (channel === "tool_done") {
      if (onToolDone) onToolDone(payload.tool, payload);
    }
  }

  window.__iw3PushEvent = handle;

  return {
    setOnDone: setOnDone,
    setOnUpdateLog: setOnUpdateLog,
    setOnUpdateDone: setOnUpdateDone,
    setOnToolLog: setOnToolLog,
    setOnToolDone: setOnToolDone,
  };
})();
