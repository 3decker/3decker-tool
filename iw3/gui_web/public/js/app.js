(function () {
  var inputPath = null;
  var outputPath = null;
  var suspended = false;

  function setRunningUi(running) {
    document.getElementById("btnStart").disabled = running;
    document.getElementById("btnCancel").disabled = !running;
    document.getElementById("btnSuspend").disabled = !running;
  }

  async function refreshPresetList() {
    var names = await IW3Api.call("list_presets");
    var list = document.getElementById("presetListOptions");
    list.innerHTML = "";
    names.forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      list.appendChild(opt);
    });
  }

  function applySettingsAndPaths(settings) {
    if (!settings) return;
    IW3Renderer.setValues(settings);
    if (settings.input) {
      inputPath = settings.input;
      document.getElementById("inputPath").value = settings.input;
    }
    if (settings.output) {
      outputPath = settings.output;
      document.getElementById("outputPath").value = settings.output;
    }
  }

  async function boot() {
    var schema = await IW3Api.call("get_schema");
    IW3Renderer.render(schema, document.getElementById("tabRail"), document.getElementById("tabPanels"));
    await IW3Standalone.render(document.getElementById("tabRail"), document.getElementById("tabPanels"));

    await refreshPresetList();
    var session = await IW3Api.call("get_session");
    applySettingsAndPaths(session);

    document.getElementById("btnPresetMovie").addEventListener("click", function () {
      IW3Renderer.setValues(window.IW3_QUICK_PRESETS.movie);
      document.getElementById("statusText").textContent =
        "Applied preset: " + window.IW3_QUICK_PRESET_LABELS.movie;
    });
    document.getElementById("btnPresetAction").addEventListener("click", function () {
      IW3Renderer.setValues(window.IW3_QUICK_PRESETS.action);
      document.getElementById("statusText").textContent =
        "Applied preset: " + window.IW3_QUICK_PRESET_LABELS.action;
    });
    document.getElementById("btnPresetDecker").addEventListener("click", function () {
      IW3Renderer.setValues(window.IW3_QUICK_PRESETS.decker);
      document.getElementById("statusText").textContent =
        "Applied preset: " + window.IW3_QUICK_PRESET_LABELS.decker;
    });

    document.getElementById("btnLoadPreset").addEventListener("click", async function () {
      var name = document.getElementById("presetSelect").value.trim();
      if (!name) return;
      var settings = await IW3Api.call("load_preset", name);
      if (!settings) {
        document.getElementById("statusText").textContent = "No preset named \"" + name + "\"";
        return;
      }
      applySettingsAndPaths(settings);
      document.getElementById("statusText").textContent = "Loaded preset: " + name;
    });

    document.getElementById("btnSavePreset").addEventListener("click", async function () {
      var name = document.getElementById("presetSelect").value.trim();
      if (!name) {
        document.getElementById("statusText").textContent = "Type a preset name first";
        return;
      }
      var settings = IW3Renderer.getSettings();
      settings.input = inputPath;
      settings.output = outputPath;
      await IW3Api.call("save_preset", name, settings);
      await refreshPresetList();
      document.getElementById("statusText").textContent = "Saved preset: " + name;
    });

    document.getElementById("btnDeletePreset").addEventListener("click", async function () {
      var name = document.getElementById("presetSelect").value.trim();
      if (!name) return;
      await IW3Api.call("delete_preset", name);
      await refreshPresetList();
      document.getElementById("statusText").textContent = "Deleted preset: " + name;
    });

    document.getElementById("btnCopyCommand").addEventListener("click", async function () {
      var settings = IW3Renderer.getSettings();
      settings.input = inputPath || "";
      settings.output = outputPath || "";
      var cmd = await IW3Api.call("copy_command", settings);
      var copied = false;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(cmd);
          copied = true;
        }
      } catch (e) {
        copied = false;
      }
      if (copied) {
        document.getElementById("statusText").textContent = "Command copied to clipboard";
      } else {
        // Clipboard write isn't guaranteed to work in every webview context --
        // show it in the Import Command box instead so it's still usable
        // (select-all + Ctrl+C) rather than silently failing.
        document.getElementById("importCommandText").value = cmd;
        document.getElementById("importCommandBox").hidden = false;
        document.getElementById("statusText").textContent =
          "Clipboard unavailable -- command shown below, select and copy manually";
      }
    });

    document.getElementById("btnImportCommand").addEventListener("click", function () {
      document.getElementById("importCommandBox").hidden = false;
      document.getElementById("importCommandText").focus();
    });
    document.getElementById("btnImportCommandCancel").addEventListener("click", function () {
      document.getElementById("importCommandBox").hidden = true;
      document.getElementById("importCommandError").textContent = "";
    });
    document.getElementById("btnImportCommandApply").addEventListener("click", async function () {
      var text = document.getElementById("importCommandText").value;
      var result = await IW3Api.call("import_command", text);
      if (result.error) {
        document.getElementById("importCommandError").textContent = result.error;
        return;
      }
      applySettingsAndPaths(result.settings);
      document.getElementById("importCommandBox").hidden = true;
      document.getElementById("importCommandError").textContent = "";
      document.getElementById("statusText").textContent = "Imported command";
    });

    document.getElementById("btnCheckUpdates").addEventListener("click", async function () {
      document.getElementById("statusText").textContent = "Checking for updates...";
      var btn = document.getElementById("btnCheckUpdates");
      btn.disabled = true;
      try {
        var result = await IW3Api.call("check_for_updates");
        document.getElementById("updateResultsText").textContent = result.message;
        document.getElementById("updateResultsBox").hidden = false;
        document.getElementById("statusText").textContent =
          result.status === "error" ? "Check for Updates failed" :
          result.status === "up_to_date" ? "Already up to date" :
          "Updates are available upstream";
      } finally {
        btn.disabled = false;
      }
    });
    document.getElementById("btnUpdateResultsClose").addEventListener("click", function () {
      document.getElementById("updateResultsBox").hidden = true;
    });

    document.getElementById("btnRunUpdate").addEventListener("click", function () {
      document.getElementById("updateConfirmBox").hidden = false;
    });
    document.getElementById("btnUpdateConfirmNo").addEventListener("click", function () {
      document.getElementById("updateConfirmBox").hidden = true;
    });
    document.getElementById("btnUpdateConfirmYes").addEventListener("click", async function () {
      document.getElementById("updateConfirmBox").hidden = true;
      document.getElementById("updateLogText").value = "";
      document.getElementById("updateLogStatus").textContent = "Running...";
      document.getElementById("btnUpdateLogClose").disabled = true;
      document.getElementById("updateLogBox").hidden = false;
      document.getElementById("btnRunUpdate").disabled = true;
      document.getElementById("btnCheckUpdates").disabled = true;
      try {
        await IW3Api.call("run_update");
      } catch (e) {
        document.getElementById("updateLogText").value += "\n" + e;
        document.getElementById("updateLogStatus").textContent = "Error";
        document.getElementById("btnUpdateLogClose").disabled = false;
        document.getElementById("btnRunUpdate").disabled = false;
        document.getElementById("btnCheckUpdates").disabled = false;
      }
    });
    document.getElementById("btnUpdateLogClose").addEventListener("click", function () {
      document.getElementById("updateLogBox").hidden = true;
    });
    IW3Progress.setOnUpdateLog(function (line) {
      var ta = document.getElementById("updateLogText");
      ta.value += line;
      ta.scrollTop = ta.scrollHeight;
    });
    IW3Progress.setOnUpdateDone(function (payload) {
      document.getElementById("updateLogStatus").textContent =
        payload.ok ? "Update finished successfully" : "Update failed -- see the log above";
      document.getElementById("btnUpdateLogClose").disabled = false;
      document.getElementById("btnRunUpdate").disabled = false;
      document.getElementById("btnCheckUpdates").disabled = false;
      document.getElementById("statusText").textContent =
        payload.ok ? "Update finished successfully" : "Update failed";
    });

    document.getElementById("btnBrowseInput").addEventListener("click", async function () {
      var p = await IW3Api.call("browse_input");
      if (p) {
        inputPath = p;
        document.getElementById("inputPath").value = p;
      }
    });

    document.getElementById("btnBrowseOutput").addEventListener("click", async function () {
      var p = await IW3Api.call("browse_output", inputPath);
      if (p) {
        outputPath = p;
        document.getElementById("outputPath").value = p;
      }
    });

    document.getElementById("btnStart").addEventListener("click", async function () {
      if (!inputPath || !outputPath) {
        document.getElementById("statusText").textContent = "Set Input and Output first";
        return;
      }
      var settings = IW3Renderer.getSettings();
      settings.input = inputPath;
      settings.output = outputPath;
      setRunningUi(true);
      document.getElementById("statusText").textContent = "Starting...";
      try {
        await IW3Api.call("start", settings);
      } catch (e) {
        setRunningUi(false);
        document.getElementById("statusText").textContent = "Error: " + e;
      }
    });

    document.getElementById("btnCancel").addEventListener("click", async function () {
      await IW3Api.call("cancel");
      document.getElementById("statusText").textContent = "Cancelling...";
    });

    document.getElementById("btnSuspend").addEventListener("click", async function () {
      suspended = !suspended;
      if (suspended) {
        await IW3Api.call("suspend");
        document.getElementById("btnSuspend").textContent = "Resume";
      } else {
        await IW3Api.call("resume");
        document.getElementById("btnSuspend").textContent = "Suspend";
      }
    });

    IW3Progress.setOnDone(function () {
      setRunningUi(false);
      suspended = false;
      document.getElementById("btnSuspend").textContent = "Suspend";
    });
  }

  window.addEventListener("pywebviewready", boot);
})();
