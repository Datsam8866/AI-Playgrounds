@echo off
chcp 65001 > nul
echo ===== Portfolio Dashboard 手動更新 =====
pwsh -NonInteractive -ExecutionPolicy Bypass -File "%~dp0daily_update.ps1"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [錯誤] 更新失敗，請查看 daily_update.log
    pause
) else (
    echo.
    echo [完成] 已更新並推送至 GitHub Pages
    timeout /t 3 > nul
)
