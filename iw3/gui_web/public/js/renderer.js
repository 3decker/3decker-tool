// Generic schema -> DOM renderer. Builds the tab rail and every field's
// control purely from the JSON schema exported by schema_export.py -- no
// hand-written markup per control, so a field added to schema.py just
// appears here with no renderer.js changes (see ADR-081 / schema.py's own
// docstring for why this exists: 132 real CLI flags is too many to
// hand-author and keep in sync by hand).
window.IW3Renderer = (function () {
  var values = {};
  var fieldsByName = {};

  // "" (an untouched/cleared text field) and null (an unset schema default)
  // are treated as the same "nothing here" value for eq/ne comparisons --
  // matches how worker.py's build_args() treats an empty string the same
  // as an omitted/None field.
  function normalize(v) { return v === "" ? null : v; }

  function ruleMatches(rule) {
    if (!rule) return true;
    var current = normalize(values[rule.field]);
    if (rule.op === "eq") return current === normalize(rule.value);
    if (rule.op === "ne") return current !== normalize(rule.value);
    if (rule.op === "in") return Array.isArray(rule.value) && rule.value.indexOf(current) !== -1;
    return true;
  }

  function applyRules() {
    Object.keys(fieldsByName).forEach(function (name) {
      var f = fieldsByName[name];
      var row = document.getElementById("field-row-" + name);
      if (!row) return;
      var visible = ruleMatches(f.visible_if);
      row.classList.toggle("hidden", !visible);
      var input = document.getElementById("field-" + name);
      if (input) input.disabled = !ruleMatches(f.enabled_if);
    });
  }

  function buildControl(f) {
    var input;
    if (f.widget === "checkbox") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!f.default;
    } else if (f.widget === "select") {
      input = document.createElement("select");
      (f.choices || []).forEach(function (choice) {
        var opt = document.createElement("option");
        opt.value = choice;
        // ADR-093: a handful of fields (Stereo Format's "vr180"/"VR90") show
        // a real display label that differs from the internal value sent to
        // worker.py -- choice_labels carries that mapping when present.
        opt.textContent = (f.choice_labels && f.choice_labels[choice]) || choice;
        input.appendChild(opt);
      });
      if (f.default != null) input.value = f.default;
    } else {
      // combo_editable -- a free-typed field with suggested values. A plain
      // text input plus a <datalist> is the closest native HTML equivalent
      // of wx's EditableComboBox (typeable, but offers the same presets).
      input = document.createElement("input");
      input.type = "text";
      input.value = f.default != null ? String(f.default) : "";
      if (f.choices && f.choices.length) {
        var listId = "list-" + f.name;
        var list = document.createElement("datalist");
        list.id = listId;
        f.choices.forEach(function (choice) {
          var opt = document.createElement("option");
          opt.value = choice;
          opt.textContent = (f.choice_labels && f.choice_labels[choice]) || choice;
          list.appendChild(opt);
        });
        input.setAttribute("list", listId);
        input.dataset.datalistEl = "1";
        input._datalist = list;
      }
    }
    input.id = "field-" + f.name;
    input.title = f.tooltip || "";
    return input;
  }

  function readValue(f, input) {
    if (f.widget === "checkbox") return input.checked;
    return input.value;
  }

  function render(schema, tabRailEl, panelsEl) {
    values = {};
    fieldsByName = {};
    tabRailEl.innerHTML = "";
    panelsEl.innerHTML = "";

    var panelByTab = {};
    var tabLabelByKey = {};
    schema.tabs.forEach(function (tab, i) {
      var tabBtn = document.createElement("div");
      tabBtn.className = "tab-item" + (i === 0 ? " active" : "");
      tabBtn.textContent = tab.label;
      tabBtn.dataset.tab = tab.key;
      tabBtn.addEventListener("click", function () { setActiveTab(tab.key); });
      tabRailEl.appendChild(tabBtn);

      var panel = document.createElement("div");
      panel.className = "tab-panel" + (i === 0 ? " active" : "");
      panel.dataset.tab = tab.key;
      panelsEl.appendChild(panel);
      panelByTab[tab.key] = panel;
      tabLabelByKey[tab.key] = tab.label;
    });

    // One card per (tab, group) -- e.g. Processor's real "Processor" +
    // "Post-Processing" StaticBox split, or Stereo Generation's "Core" /
    // "Inpainting" / etc. clusters. A field with no group (schema.py's
    // FIELD_GROUPS has no entry for it) falls back to the tab's own label,
    // so a tab nobody grouped still renders as today: one box.
    var cardByTabGroup = {};
    function cardFor(tabKey, groupLabel) {
      var key = tabKey + "::" + groupLabel;
      if (cardByTabGroup[key]) return cardByTabGroup[key];
      var card = document.createElement("div");
      card.className = "card";
      var h3 = document.createElement("h3");
      h3.textContent = groupLabel;
      card.appendChild(h3);
      panelByTab[tabKey].appendChild(card);
      cardByTabGroup[key] = card;
      return card;
    }

    schema.fields.forEach(function (f) {
      fieldsByName[f.name] = f;

      var row = document.createElement("div");
      row.className = "field-row";
      row.id = "field-row-" + f.name;

      var label = document.createElement("label");
      label.className = "field-label";
      label.textContent = f.label;
      label.title = f.tooltip || "";
      row.appendChild(label);

      var controlWrap = document.createElement("div");
      controlWrap.className = "field-control";
      var input = buildControl(f);
      controlWrap.appendChild(input);
      if (input._datalist) controlWrap.appendChild(input._datalist);
      row.appendChild(controlWrap);

      // Read the REAL initial DOM value, not f.default directly -- for a
      // "select" with no explicit default, the browser auto-selects the
      // first <option> (e.g. Device defaulting to the first real GPU), and
      // values must track what's actually shown or getSettings() silently
      // disagrees with the visible UI (a real bug caught via live testing:
      // the Device dropdown visually showed a GPU selected but
      // getSettings() returned null for it until the user touched it).
      values[f.name] = readValue(f, input);

      input.addEventListener("change", function () {
        values[f.name] = readValue(f, input);
        applyRules();
      });
      input.addEventListener("input", function () {
        values[f.name] = readValue(f, input);
      });

      var groupLabel = f.group || tabLabelByKey[f.tab];
      var targetCard = cardFor(f.tab, groupLabel);
      targetCard.appendChild(row);
    });

    applyRules();
  }

  function setActiveTab(key) {
    document.querySelectorAll(".tab-item").forEach(function (el) {
      el.classList.toggle("active", el.dataset.tab === key);
    });
    document.querySelectorAll(".tab-panel").forEach(function (el) {
      el.classList.toggle("active", el.dataset.tab === key);
    });
  }

  function getSettings() {
    return Object.assign({}, values);
  }

  // Applies a settings object (from a Quick Preset, a loaded named preset,
  // Import Command, or the restored last session) back onto the real form
  // controls -- not just the internal `values` map, so what's on screen
  // always matches what getSettings() would report next. Unknown keys
  // (e.g. a preset saved before a field existed) and undefined values are
  // silently skipped rather than raising, since presets/sessions are
  // expected to drift from the current schema over time.
  function setValues(obj) {
    if (!obj) return;
    Object.keys(obj).forEach(function (name) {
      var f = fieldsByName[name];
      var input = document.getElementById("field-" + name);
      if (!f || !input) return;
      var value = obj[name];
      if (value === undefined) return;
      if (f.widget === "checkbox") {
        input.checked = !!value;
      } else {
        input.value = value == null ? "" : String(value);
      }
      values[name] = readValue(f, input);
    });
    applyRules();
  }

  return { render: render, getSettings: getSettings, setValues: setValues };
})();
