@echo off
@rem 3DECKER setup -- downloads and installs everything needed to run 3DECKER
@rem from scratch: Python, the source code, the video tools, and the AI
@rem models. This is a thin wrapper around setup.ps1 (the real logic) that
@rem exists to sidestep two real Windows quirks a direct "Run with
@rem PowerShell" on setup.ps1 can hit:
@rem   1) PowerShell's execution policy can silently refuse to run ANY
@rem      downloaded .ps1 file at all -- batch files aren't subject to that
@rem      restriction, so this wrapper can't be blocked the same way.
@rem   2) A bare "Run with PowerShell" closes the window the instant the
@rem      script ends, success or failure, before you can read anything.
@rem      setup.ps1 itself now pauses before closing in the normal case;
@rem      this wrapper adds its own pause for the one failure mode that
@rem      happens before setup.ps1's own code ever gets a chance to run.
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

exit /b 0

:on_error
echo.
echo Setup did not complete -- scroll up to see what went wrong.
pause
exit /b 1
