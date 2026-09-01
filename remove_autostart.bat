@echo off
title Remove Project Aegis Autostart
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\AegisTerminal.bat"

if exist "%SHORTCUT_PATH%" (
    del "%SHORTCUT_PATH%"
    echo Project Aegis autostart removed from Windows Startup.
) else (
    echo Autostart was not active.
)
pause