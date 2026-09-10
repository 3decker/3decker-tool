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
        opt.textContent = choice;
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
      var card = document.createElement("div");
      card.className = "card";
      var h3 = document.createElement("h3");
      h3.textContent = tab.label;
      card.appendChild(h3);
      panel.appendChild(card);
      panelsEl.appendChild(panel);
      panelByTab[tab.key] = card;
    });

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

      var targetCard = panelByTab[f.tab];
      if (targetCard) targetCard.appendChild(row);
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

  return { render: render, getSettings: getSettings };
})();
