(function () {
  var inputPath = null;
  var outputPath = null;
  var suspended = false;

  function setRunningUi(running) {
    document.getElementById("btnStart").disabled = running;
    document.getElementById("btnCancel").disabled = !running;
    document.getElementById("btnSuspend").disabled = !running;
  }

  async function boot() {
    var schema = await IW3Api.call("get_schema");
    IW3Renderer.render(schema, document.getElementById("tabRail"), document.getElementById("tabPanels"));

    document.getElementById("btnBrowseInput").addEventListener("click", async function () {
      var p = await IW3Api.call("browse_input");
      if (p) {
        inputPath = p;
        document.getElementById("inputPath").value = p;
      }
    });

    document.getElementById("btnBrowseOutput").addEventListener("click", async function () {
      var p = await IW3Api.call("browse_output");
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
