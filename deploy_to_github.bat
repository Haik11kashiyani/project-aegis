@echo off
title Deploy Project Aegis to GitHub
cd /d "%~dp0"

echo ========================================================
echo   PUSHING PROJECT AEGIS v4 TO GITHUB
echo ========================================================
echo.

git push origin main

echo.
if %ERRORLEVEL% equ 0 (
    echo ========================================================
    echo   SUCCESS! All code is now live on GitHub!
    echo   Repository: https://github.com/Haik11kashiyani/project-aegis
    echo ========================================================
) else (
    echo ========================================================
    echo   Push encountered an error. Please check your GitHub login.
    echo ========================================================
)
pause