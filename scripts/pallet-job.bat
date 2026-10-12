@echo off
rem Double-click on Windows: runs the whole overnight job (or pass arguments, e.g. run --smoke).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pallet-job.ps1" %*
echo.
echo Finished (exit code %ERRORLEVEL%).
pause
