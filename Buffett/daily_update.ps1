# daily_update.ps1 — Portfolio Dashboard 自動更新
# 每日由 Windows Task Scheduler 觸發，更新報價 + 重建 Dashboard + push GitHub

$ErrorActionPreference = "Stop"
$LogFile = "$PSScriptRoot\daily_update.log"
$PythonExe = "python"
$RepoRoot = "C:\Users\dat_s\OneDrive\Documents\AI Playgrounds"
$BuffettDir = "$RepoRoot\Buffett"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $LogFile -Append
}

Log "===== 開始每日更新 ====="

try {
    Set-Location $BuffettDir

    Log "▶ update_portfolio.py"
    & $PythonExe -X utf8 update_portfolio.py
    if ($LASTEXITCODE -ne 0) { throw "update_portfolio.py 失敗 (exit $LASTEXITCODE)" }

    Log "▶ generate_portfolio_dashboard.py"
    & $PythonExe -X utf8 generate_portfolio_dashboard.py
    if ($LASTEXITCODE -ne 0) { throw "generate_portfolio_dashboard.py 失敗 (exit $LASTEXITCODE)" }

    Set-Location $RepoRoot
    Log "▶ git add + commit + push"
    git add Buffett/portfolio_dashboard.html
    $today = Get-Date -Format "yyyy-MM-dd"
    git commit -m "每日更新 Dashboard $today" 2>&1 | ForEach-Object { Log $_ }
    git push origin main 2>&1 | ForEach-Object { Log $_ }

    Log "✅ 完成"
} catch {
    Log "❌ 錯誤：$_"
    exit 1
}
