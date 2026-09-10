# 3DECKER — How to Install

This is the dedicated install guide for **3DECKER** (this fork's customized build
of iw3/nunif). For what 3DECKER actually does and how to use it once installed,
see `3DECKER_README.md`. For a full reference of every individual setting, see
`3DECKER_SETTINGS_GUIDE.md`.

---

## Before you start

- **Windows 7 or later** (Windows 10/11 recommended). Windows Server 2008 R2 and
  later also works.
- **Microsoft Visual C++ Redistributable** — [download here](https://aka.ms/vc14/vc_redist.x64.exe)
  if you don't already have it. Most Windows PCs already do (lots of software
  depends on it), but if the app fails to start at all, this is the first thing
  to check.
- **An NVIDIA, AMD, or Intel GPU is recommended** for real speed — the app also
  runs on CPU only, but dramatically slower. NVIDIA is the most tested path.
- **Enable long file paths** (optional, but recommended): Windows normally limits
  file paths to 260 characters, which can cause `No module named '...'`-style
  errors if this project ends up in a deeply nested folder. Fix once, system-wide:
  download and run [`enable_long_path.reg`](https://raw.githubusercontent.com/3decker/3decker-tool/my-customizations/windows_package/torch_compile/enable_long_path.reg)
  (needs administrator rights) and reboot. See Microsoft's own
  [Maximum Path Length Limitation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)
  page for the full explanation.

---

## Choose your install path

Pick whichever matches your situation — both end up at the same place, a working
copy you launch with `iw3-gui.bat`.

| | Option 1: Fresh install | Option 2: You already have iw3/nunif |
|---|---|---|
| Use this if... | You don't have Python/the tools/the AI models set up yet | You already have a working iw3/nunif install (this fork, an earlier version, or plain upstream nunif) |
| What it does | Downloads and sets up everything from scratch | Switches your existing source code over to 3DECKER's customized version |
| Re-downloads models/tools? | Yes — one-time, several GB | No — nothing large is re-downloaded |

### Option 1: Fresh install

Two small files download and set up everything: Python, the source code, the
video tools (ffmpeg, MKVToolNix, dovi_tool, hdr10plus_tool), and the AI models.

1. Go to the [`windows_package`](https://github.com/3decker/3decker-tool/tree/my-customizations/windows_package)
   folder in the repo and download **both** `setup.bat` and `setup.ps1` (open
   each file, then use the download button or the **Raw** button → Save As).
   Put them together, by themselves, in an empty folder — this becomes your
   install folder.
2. Right-click **each** file → **Properties** → if you see an **Unblock**
   checkbox, check it → **OK**. (Windows flags anything downloaded from a
   browser this way; unblocking lets it run.)
3. Double-click **`setup.bat`** to run it. (Use this one, not `setup.ps1`
   directly — `setup.bat` sidesteps a real Windows quirk where PowerShell's
   own security settings can silently block `setup.ps1` from running at all
   when launched directly, with the window just closing instantly and no
   clear error. `setup.bat` isn't affected by that, and its window always
   stays open to show you what happened, success or failure.)
4. It automatically detects your GPU and installs the matching PyTorch build
   (works for RTX 50-series/Blackwell cards and older NVIDIA generations alike
   — you don't need to know which one you need). If you have an AMD or Intel
   GPU instead, run it from a terminal instead with `setup.bat -TorchVariant
   rocm` or `setup.bat -TorchVariant xpu`.
5. This downloads several GB (Python packages + AI models), so it takes a
   while depending on your connection — the script prints its progress at each
   step. When it prints **"Setup complete,"** press Enter to close the window,
   then launch `iw3-gui.bat` from that same folder.

**If you'd rather download the whole repository as a ZIP** (GitHub's green
"Code" → "Download ZIP" button) instead of just those two files:
1. Extract the ZIP. You'll get a folder named something like
   `3decker-tool-my-customizations`.
2. Rename that folder to `nunif`.
3. Create a new empty folder (this becomes your install folder) and move the
   renamed `nunif` folder inside it.
4. Copy both `nunif\windows_package\setup.bat` and
   `nunif\windows_package\setup.ps1` up one level, next to (not inside) the
   `nunif` folder.
5. Run `setup.bat` from there — same as step 3 above. It'll detect the source
   is already present and skip straight to installing everything around it.
   One tradeoff: a ZIP download has no git history, so later updates and the
   in-app "Run Update" button won't be able to auto-pull new versions — if you
   want that, use the direct two-file method instead.

If something fails partway through (a download hiccup, etc.), it's safe to just
run `setup.bat` again — it skips anything already done and only retries what's
missing.

### Option 2: You already have a working iw3/nunif install

You don't need to redownload Python, the tools, or the AI models — just switch
your existing `nunif` folder's source code over to the 3DECKER-customized
version:

1. Open `nunif-prompt.bat` (or any terminal) and go into your existing `nunif`
   folder.
2. Run:
   ```
   git remote add 3decker https://github.com/3decker/3decker-tool.git
   git fetch 3decker
   git checkout -b my-customizations 3decker/my-customizations
   ```
3. Done — your Python, tools, and downloaded AI models stay exactly where they
   are; nothing gets re-downloaded. Only the source code (the GUI and all its
   features) switches over. Launch `iw3-gui.bat` as usual.

This only works if your existing `nunif` folder is a real git clone — true for
anyone who used the standard nunif-windows install/update process. To go back
to the plain, non-customized version later, run `git checkout master` in that
same folder.

---

## After installing

- **Launch it:** `iw3-gui.bat` (the 3D converter) or `waifu2x-gui.bat` (the
  image upscaler) in your install folder. The first launch can take a little
  longer than normal while things warm up.
- **torch.compile (optional, faster conversions):** needs one extra one-time
  setup step beyond what `setup.ps1` installs automatically. See
  `windows_package/docs/torch_compile.md` for exactly what's needed.
- **Updating later:** if you installed via a real git clone (Option 1's direct
  `setup.ps1` route, or Option 2), the in-app **Run Update** button (top
  toolbar) pulls the latest source. From a ZIP-download install, delete the
  `nunif` folder and re-run `setup.ps1` to get a real git clone instead.
- **Uninstalling:** delete the entire install folder. Nothing this project
  installs lives anywhere else on your system.

## If something goes wrong

- **A black window flashes and disappears, or Windows Defender SmartScreen
  blocks the file:** right-click the file → Properties → check **Unblock** →
  OK, then try again. If SmartScreen still blocks it, click **More info**,
  then **Run anyway**.
- **`No module named '...'` errors:** almost always the long-path issue — see
  "Before you start" above.
- **`CUDA error: no kernel image is available for execution on the device`:**
  your PyTorch build doesn't match your GPU. If you used `setup.ps1`, re-run it
  — auto-detection should pick the right build; if it still happens, pass the
  variant explicitly (e.g. `-TorchVariant cu130` for RTX 50-series cards).
