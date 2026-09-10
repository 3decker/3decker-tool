// Renders the Standalone Tools tab: one card per tool, fields built
// generically from each tool's real create_parser() (via
// api.get_tool_schemas()) -- same "schema is the source of truth, not
// hand-authored markup" principle as the main renderer.js, just without
// that renderer's tab/group/visible_if machinery, which these small,
// mostly-independent tool forms don't need.
window.IW3Standalone = (function () {
  // Path-like fields get a Browse button. "output" is the only Save-dialog
  // field across all 6 tools' real argparse definitions (reinject_hdr_cli,
  // subtitle_mux_cli, audio_mux_cli, stereo_mode_tag_cli, sharpen_cli,
  // rife_cli) -- everything else path-shaped is an Open dialog.
  var OPEN_PATH_FIELDS = new Set(["input", "source", "converted", "srt", "audio", "rife_manifest"]);
  var SAVE_PATH_FIELDS = new Set(["output"]);

  function buildFieldInput(toolKey, f) {
    var input;
    if (f.widget === "checkbox") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!f.default;
    } else if (f.widget === "select") {
      input = document.createElement("select");
      (f.choices || []).forEach(function (c) {
        var opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        input.appendChild(opt);
      });
      if (f.default != null) input.value = f.default;
    } else {
      // combo_editable equivalent -- a free-typed field with suggested
      // values via a <datalist>, same pattern as renderer.js's own
      // combo_editable widget (a real gap fixed in ADR-089: this builder
      // previously built a plain text input with no datalist at all, so
      // TOOL_FIELD_CHOICES entries added on the backend had nowhere to
      // show up on screen).
      input = document.createElement("input");
      input.type = "text";
      input.value = f.default != null ? String(f.default) : "";
      if (f.choices && f.choices.length) {
        var listId = "tool-list-" + toolKey + "-" + f.name;
        var list = document.createElement("datalist");
        list.id = listId;
        f.choices.forEach(function (c) {
          var opt = document.createElement("option");
          opt.value = c;
          list.appendChild(opt);
        });
        input.setAttribute("list", listId);
        input._datalist = list;
      }
    }
    input.id = "tool-" + toolKey + "-" + f.name;
    input.title = f.help || "";
    return input;
  }

  function readFieldValue(f, input) {
    if (f.widget === "checkbox") return input.checked;
    return input.value;
  }

  async function browseFor(toolKey, f, input) {
    if (SAVE_PATH_FIELDS.has(f.name)) {
      var p = await IW3Api.call("browse_save_file", null, input.value || "");
      if (p) input.value = p;
    } else {
      var p2 = await IW3Api.call("browse_open_file", null);
      if (p2) input.value = p2;
    }
  }

  function buildToolCard(toolKey, tool, container) {
    var card = document.createElement("div");
    card.className = "card";
    var h3 = document.createElement("h3");
    h3.textContent = tool.label;
    card.appendChild(h3);

    tool.fields.forEach(function (f) {
      var row = document.createElement("div");
      row.className = "field-row";

      var label = document.createElement("label");
      label.className = "field-label";
      label.textContent = f.label + (f.required ? " *" : "");
      label.title = f.help || "";
      row.appendChild(label);

      var controlWrap = document.createElement("div");
      controlWrap.className = "field-control tool-field-control";
      var input = buildFieldInput(toolKey, f);
      controlWrap.appendChild(input);
      if (input._datalist) controlWrap.appendChild(input._datalist);

      if (OPEN_PATH_FIELDS.has(f.name) || SAVE_PATH_FIELDS.has(f.name)) {
        var browseBtn = document.createElement("button");
        browseBtn.className = "btn";
        browseBtn.textContent = "Browse";
        browseBtn.type = "button";
        browseBtn.addEventListener("click", function () { browseFor(toolKey, f, input); });
        controlWrap.appendChild(browseBtn);
      }

      row.appendChild(controlWrap);
      card.appendChild(row);
    });

    var actions = document.createElement("div");
    actions.className = "tool-actions";
    var runBtn = document.createElement("button");
    runBtn.className = "btn primary";
    runBtn.textContent = "Run";
    runBtn.id = "tool-" + toolKey + "-run";
    var clearBtn = document.createElement("button");
    clearBtn.className = "btn";
    clearBtn.textContent = "Clear Log";
    clearBtn.type = "button";
    actions.appendChild(runBtn);
    actions.appendChild(clearBtn);
    card.appendChild(actions);

    var log = document.createElement("textarea");
    log.className = "tool-log";
    log.id = "tool-" + toolKey + "-log";
    log.rows = 6;
    log.readOnly = true;
    card.appendChild(log);

    runBtn.addEventListener("click", async function () {
      var values = {};
      tool.fields.forEach(function (f) {
        var input = document.getElementById("tool-" + toolKey + "-" + f.name);
        values[f.name] = readFieldValue(f, input);
      });
      log.value = "";
      runBtn.disabled = true;
      try {
        await IW3Api.call("run_tool", toolKey, values);
      } catch (e) {
        log.value += "\n" + e;
        runBtn.disabled = false;
      }
    });
    clearBtn.addEventListener("click", function () { log.value = ""; });

    container.appendChild(card);
  }

  async function render(tabRailEl, panelsEl) {
    var schemas = await IW3Api.call("get_tool_schemas");

    var tabBtn = document.createElement("div");
    tabBtn.className = "tab-item";
    tabBtn.textContent = "Standalone Tools";
    tabBtn.dataset.tab = "standalone_tools";
    tabBtn.addEventListener("click", function () {
      document.querySelectorAll(".tab-item").forEach(function (el) {
        el.classList.toggle("active", el === tabBtn);
      });
      document.querySelectorAll(".tab-panel").forEach(function (el) {
        el.classList.toggle("active", el.dataset.tab === "standalone_tools");
      });
    });
    tabRailEl.appendChild(tabBtn);

    var panel = document.createElement("div");
    panel.className = "tab-panel";
    panel.dataset.tab = "standalone_tools";
    panelsEl.appendChild(panel);

    Object.keys(schemas).forEach(function (toolKey) {
      buildToolCard(toolKey, schemas[toolKey], panel);
    });

    IW3Progress.setOnToolLog(function (toolKey, line) {
      var log = document.getElementById("tool-" + toolKey + "-log");
      if (log) { log.value += line; log.scrollTop = log.scrollHeight; }
    });
    IW3Progress.setOnToolDone(function (toolKey, payload) {
      var runBtn = document.getElementById("tool-" + toolKey + "-run");
      if (runBtn) runBtn.disabled = false;
      var log = document.getElementById("tool-" + toolKey + "-log");
      if (log) log.value += "\n" + (payload.ok ? "Finished successfully." : "Failed -- see log above.");
    });
  }

  return { render: render };
})();
