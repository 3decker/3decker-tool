@echo off

setlocal enabledelayedexpansion
@rem This script lives in windows_package\, two levels below the real distribution
@rem root (root\nunif\windows_package\) -- unlike update.bat (which lives AT the
@rem root, where %~dp0\setenv.bat is correct), a bare %~dp0\setenv.bat here would
@rem silently call THIS folder's own template copy of setenv.bat, which computes
@rem ROOT_DIR/NUNIF_DIR/MINGIT_DIR etc. relative to windows_package\ instead of the
@rem real root -- pointing PATH at a git\cmd that doesn't exist there and failing
@rem with "'git' is not recognized" (confirmed live, this was a real, shipped bug).
call "%~dp0..\..\setenv.bat"

@rem check to make sure the variables are available
if "%ROOT_DIR%"=="" goto :on_error
if "%PYTHON_DIR%"=="" goto :on_error
if "%NUNIF_DIR%"=="" goto :on_error

@rem This script only ever runs from inside an already-working GUI, so python/git
@rem are already known-good -- unlike update.bat, no install_python/install_git
@rem bootstrap branches are needed here.

@rem Pull from THIS fork's own repo/branch specifically -- hardcoded, never
@rem inferred from whatever the local branch happens to track (that's the whole
@rem point of this script existing separately from update.bat: see
@rem docs/ai/AI_DECISIONS.md for the ADR covering why). Keep this URL/branch in
@rem sync with THIS_FORK_REPO_URL/THIS_FORK_BRANCH in iw3/update_check.py.
git -C "%NUNIF_DIR%" fetch https://github.com/3decker/3decker-tool.git my-customizations
if %ERRORLEVEL% neq 0 goto :on_error
git -C "%NUNIF_DIR%" merge --ff-only FETCH_HEAD
if %ERRORLEVEL% neq 0 (
  git -C "%NUNIF_DIR%" reset --hard FETCH_HEAD
  if !ERRORLEVEL! neq 0 goto :on_error
)

@rem update update-installer.bat (same as update.bat's own convention) -- must land
@rem at the real root, not this folder, same %~dp0 reasoning as above.
copy /y "%NUNIF_DIR%\windows_package\update-installer.bat" "%~dp0..\..\update-installer.bat"

@rem ADR-117: also refresh the launcher .bat files at the root directly, instead of
@rem only refreshing update-installer.bat and leaving the user to separately
@rem double-click it by hand -- confirmed live this was a real gap (3decker-gui.bat
@rem never appeared at the root after "Install Update Now" until update-installer.bat
@rem was run manually afterward). Same copy list update-installer.bat uses, inlined
@rem here without its own pause/exit so this runs unattended as part of the update.
copy /y "%NUNIF_DIR%\windows_package\setenv.bat" "%~dp0..\..\setenv.bat"
copy /y "%NUNIF_DIR%\windows_package\update.bat" "%~dp0..\..\update.bat"
copy /y "%NUNIF_DIR%\windows_package\install.bat" "%~dp0..\..\install.bat"
copy /y "%NUNIF_DIR%\windows_package\nunif-prompt.bat" "%~dp0..\..\nunif-prompt.bat"
copy /y "%NUNIF_DIR%\windows_package\3decker-gui.bat" "%~dp0..\..\3decker-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\iw3-desktop-gui.bat" "%~dp0..\..\iw3-desktop-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\iw3-player-gui.bat" "%~dp0..\..\iw3-player-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\waifu2x-gui.bat" "%~dp0..\..\waifu2x-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\waifu2x-web.bat" "%~dp0..\..\waifu2x-web.bat"
xcopy "%NUNIF_DIR%\windows_package\torch_compile" "%~dp0..\..\torch_compile" /E /H /Y /I
if exist "%~dp0..\..\torch_compile\install_triton_windows.bat" del /f /q "%~dp0..\..\torch_compile\install_triton_windows.bat"

echo Installing Python Packages...
call :pip_retry --upgrade pip
if %ERRORLEVEL% neq 0 goto :on_error
call :pip_retry --upgrade -r "%NUNIF_DIR%\requirements-torch.txt"
if %ERRORLEVEL% neq 0 goto :on_error
call :pip_retry --upgrade -r "%NUNIF_DIR%\requirements.txt"
if %ERRORLEVEL% neq 0 goto :on_error
call :pip_retry --upgrade -r "%NUNIF_DIR%\requirements-gui.txt"
if %ERRORLEVEL% neq 0 goto :on_error


echo Download Models...
pushd "%NUNIF_DIR%" && python -m waifu2x.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error

pushd "%NUNIF_DIR%" && python -m iw3.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error

@rem ADR-137: registers the 3 optional Aether inpaint models if they aren't
@rem already registered -- covers installs that completed setup.ps1 BEFORE
@rem this feature existed (setup.ps1 only ever runs once, so those installs
@rem would otherwise never get this file no matter how many updates they run).
@rem No-ops if iw3/inpaint_models.yml already exists (see
@rem ensure_optional_inpaint_models_registered() in iw3/inpaint_utils.py).
pushd "%NUNIF_DIR%" && python -m iw3.register_optional_inpaint_models && popd
if %ERRORLEVEL% neq 0 goto :on_error

@rem ADR-182: installs the 3D Blu-ray import tools (patched edge264-mvc decoder +
@rem tsMuxeR) if missing. Deliberately NOT fatal -- only the optional 3D Blu-ray
@rem import needs them, so a failed download must not fail the whole update.
pushd "%NUNIF_DIR%" && python -m iw3.install_mvc_tools & popd


@rem warmup, create pyc
pushd "%NUNIF_DIR%" && python -m iw3.gui --help > nul && popd
if %ERRORLEVEL% neq 0 goto :on_error
pushd "%NUNIF_DIR%" && python -m waifu2x.gui --help > nul && popd
if %ERRORLEVEL% neq 0 goto :on_error


@rem all succeeded
echo Successfully installed 3DECKER update
exit /b 0


@rem Runs "pip install --no-cache-dir <args>" and retries up to 3 times if it fails.
@rem Real user report: one dropped GitHub connection ("Connection was reset" while
@rem pip cloned a git-based requirement) failed the whole update. ping is used as the
@rem delay because `timeout` errors out when there is no console (the GUI runs this
@rem with no keyboard attached).
:pip_retry
  set "PIP_TRY=1"
:pip_retry_loop
  python -m pip install --no-cache-dir %*
  if !ERRORLEVEL! equ 0 exit /b 0
  if !PIP_TRY! geq 3 exit /b 1
  set /a PIP_TRY+=1
  echo Download failed - probably a connection problem. Retrying in 10 seconds (attempt !PIP_TRY! of 3^)...
  ping -n 11 127.0.0.1 > nul
  goto :pip_retry_loop

@rem ADR-143: real, user-reported confusion -- both this and the :on_error path
@rem below used to end with `pause`. This script only ever runs unattended from
@rem inside the GUI (see the note at the top of this file), whose RunUpdateDialog
@rem is a read-only log window with no way to forward a keypress to this process.
@rem Live-tested this directly (matching gui.py's exact subprocess.Popen(...,
@rem stdin=subprocess.DEVNULL) invocation, launched from pythonw.exe with no
@rem inherited console -- the same conditions the real GUI runs under): `pause`
@rem does NOT actually hang the process here -- it exits immediately, and the
@rem update genuinely finished (the log window's Close button really was already
@rem enabled). The real bug is misleading leftover text: "Press any key to
@rem continue . . ." stays visible in the log after the update already succeeded,
@rem and pressing a key in that read-only window is a no-op no matter what --
@rem a real user reported exactly this, describing it as "stuck" and only
@rem discovering Close was the correct (and already working) action. Removed both
@rem `pause` calls entirely -- nothing here was ever meant to be read from a real
@rem interactive console, so the prompt never served a purpose.
:on_error
  echo Error!
  exit /b 1
