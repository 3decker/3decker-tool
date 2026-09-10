@echo off
@rem 3DECKER setup -- downloads and installs everything needed to run 3DECKER
@rem from scratch: Python, the source code, the video tools, and the AI
@rem models. This is a thin wrapper around setup.ps1 (the real logic) that
@rem exists to sidestep a real Windows quirk a direct "Run with PowerShell"
@rem on setup.ps1 can hit: PowerShell's execution policy can silently
@rem refuse to run ANY downloaded .ps1 file at all -- batch files aren't
@rem subject to that restriction, so this wrapper can't be blocked the
@rem same way. This window also always pauses before closing, success or
@rem failure, matching update.bat/install.bat's own convention -- setup.ps1
@rem itself does NOT pause on its own (a second, separate pause inside it
@rem caused a real, confirmed bug: leftover console input collided with
@rem this wrapper's own next command once control returned to it).
@rem
@rem Save this file AND setup.ps1 together, by themselves, in an empty
@rem folder, and run this one. That folder becomes your install location.
@rem See 3DECKER_INSTALL.md for full details.

setlocal
set SCRIPT_DIR=%~dp0

if not exist "%SCRIPT_DIR%setup.ps1" (
  echo ERROR: setup.ps1 was not found next to this file.
  echo Download setup.ps1 from the same place you got this .bat file, put
  echo it in this same folder, then run this again.
  goto :on_error
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup.ps1" %*
if %ERRORLEVEL% neq 0 goto :on_error

echo.
echo Setup complete -- see the output above. You can now launch iw3-gui.bat or waifu2x-gui.bat in this folder.
pause
exit /b 0

:on_error
echo.
echo Setup did not complete -- scroll up to see what went wrong.
pause
exit /b 1
