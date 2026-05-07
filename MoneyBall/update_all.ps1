# MoneyBall — 每日比賽資料更新腳本
# 呼叫方式: powershell -File update_all.ps1
# 建議排程: 每天 14:00 & 22:00

$ErrorActionPreference = "Continue"
$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Year = (Get-Date).Year
$LogDir = "$BaseDir\logs"
$LogFile = "$LogDir\update_$(Get-Date -Format 'yyyy-MM-dd').log"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts  $msg" | Tee-Object -FilePath $LogFile -Append
}

Log "===== 開始更新 year=$Year ====="

# --- MLB ---
Log "[MLB] 開始"
Set-Location "$BaseDir\MLB"
$out = python mlb_boxscore_scraper.py --start-year $Year --end-year $Year 2>&1
Log "[MLB] $out"

# --- KBO ---
Log "[KBO] 開始"
Set-Location "$BaseDir\KBO"
$out = python kbo_boxscore_scraper.py --start-year $Year --end-year $Year 2>&1
Log "[KBO] $out"

# --- CPBL ---
Log "[CPBL] 開始"
Set-Location "$BaseDir"
for ($i = 14; $i -ge 0; $i--) {
    $TargetDate = (Get-Date).Date.AddDays(-$i).ToString("yyyy-MM-dd")
    $out = python playsport_results_sync.py --league cpbl --date $TargetDate 2>&1
    Log "[CPBL $TargetDate] $out"
}

# --- NPB (2 步驟: schedule → boxscore) ---
Log "[NPB] schedule 開始"
Set-Location "$BaseDir\NPB"
$out = python npb_schedule_scraper.py --year $Year 2>&1
Log "[NPB schedule] $out"

Log "[NPB] boxscore 開始"
$out = python npb_boxscore_scraper.py --year $Year 2>&1
Log "[NPB boxscore] $out"

Log "===== 更新完成 ====="
Set-Location $BaseDir
