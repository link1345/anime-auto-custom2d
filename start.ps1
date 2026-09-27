param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11以上をインストールしてください。' }
}
& .\.venv\Scripts\python.exe -c "import aiohttp, PIL, numpy"
if ($LASTEXITCODE -ne 0) {
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw '依存パッケージの導入に失敗しました。' }
}
Write-Host "Nagi Studio: http://127.0.0.1:$Port"
& .\.venv\Scripts\python.exe -m src.server --port $Port
