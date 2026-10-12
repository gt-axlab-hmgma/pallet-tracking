@echo off
rem Double-click on Windows: the short check (a few minutes) that this machine can run the job.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pallet-job.ps1" run --smoke %*
echo.
echo Finished (exit code %ERRORLEVEL%).
pause
