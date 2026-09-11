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
  if %ERRORLEVEL% neq 0 goto :on_error
)

@rem update update-installer.bat (same as update.bat's own convention) -- must land
@rem at the real root, not this folder, same %~dp0 reasoning as above.
copy /y "%NUNIF_DIR%\windows_package\update-installer.bat" "%~dp0..\..\update-installer.bat"

echo Installing Python Packages...
python -m pip install --no-cache-dir --upgrade pip
if %ERRORLEVEL% neq 0 goto :on_error
python -m pip install --no-cache-dir --upgrade -r "%NUNIF_DIR%\requirements-torch.txt"
if %ERRORLEVEL% neq 0 goto :on_error
python -m pip install --no-cache-dir --upgrade -r "%NUNIF_DIR%\requirements.txt"
if %ERRORLEVEL% neq 0 goto :on_error
python -m pip install --no-cache-dir --upgrade -r "%NUNIF_DIR%\requirements-gui.txt"
if %ERRORLEVEL% neq 0 goto :on_error


echo Download Models...
pushd "%NUNIF_DIR%" && python -m waifu2x.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error

pushd "%NUNIF_DIR%" && python -m iw3.download_models && popd
if %ERRORLEVEL% neq 0 goto :on_error


@rem warmup, create pyc
pushd "%NUNIF_DIR%" && python -m iw3.gui --help > nul && popd
if %ERRORLEVEL% neq 0 goto :on_error
pushd "%NUNIF_DIR%" && python -m waifu2x.gui --help > nul && popd
if %ERRORLEVEL% neq 0 goto :on_error


@rem all succeeded
echo Successfully installed 3DECKER update
pause
exit /b 0


:on_error
  echo Error!
  pause
  exit /b 1
