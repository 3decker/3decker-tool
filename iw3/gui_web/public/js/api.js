// Thin wrapper around pywebview.api.* -- confirmed in the Phase 0 spike that
// window.pywebview.api is only attached asynchronously after the page's own
// 'pywebviewready' event, so every call here waits for that first instead
// of assuming it's already there.
window.IW3Api = (function () {
  function waitReady(timeoutMs) {
    timeoutMs = timeoutMs || 8000;
    return new Promise(function (resolve, reject) {
      if (window.pywebview && window.pywebview.api) { resolve(); return; }
      var start = Date.now();
      var timer = setInterval(function () {
        if (window.pywebview && window.pywebview.api) {
          clearInterval(timer);
          resolve();
        } else if (Date.now() - start > timeoutMs) {
          clearInterval(timer);
          reject(new Error("pywebview API never became ready"));
        }
      }, 50);
    });
  }

  async function call(method) {
    var args = Array.prototype.slice.call(arguments, 1);
    await waitReady();
    return window.pywebview.api[method].apply(window.pywebview.api, args);
  }

  return { call: call };
})();
