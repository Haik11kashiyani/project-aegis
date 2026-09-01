@echo off
title Setup Project Aegis 24/7 Windows Autostart
cd /d "%~dp0"

echo ========================================================
echo   PROJECT AEGIS - 24/7 WINDOWS AUTOSTART INSTALLER
echo ========================================================
echo.
echo This will configure Project Aegis to automatically boot
echo in the background whenever your Windows computer turns on.
echo You will NEVER need to double-click scripts manually again!
echo.

set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\AegisTerminal.bat"

echo Creating autostart trigger in:
echo "%SHORTCUT_PATH%"

(
  echo @echo off
  echo cd /d "%~dp0"
  echo call "%~dp0start_terminal.bat"
) > "%SHORTCUT_PATH%"

echo.
echo ========================================================
echo   SUCCESS! 24/7 AUTOSTART IS NOW ACTIVE!
echo   Whenever your PC boots up, Aegis will run
echo   automatically and begin trading and self-evolving.
echo ========================================================
pause