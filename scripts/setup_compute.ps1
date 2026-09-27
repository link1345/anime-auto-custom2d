param([string]$Python = 'python', [string]$TorchIndex = 'https://download.pytorch.org/whl/cu128')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path -LiteralPath '.venv-inference\Scripts\python.exe')) {
    & $Python -m venv .venv-inference
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12で仮想環境を作成してください。' }
}
$ComputePython = Join-Path (Get-Location) '.venv-inference\Scripts\python.exe'
& $ComputePython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch導入に失敗しました。' }
& $ComputePython -m pip install mediapipe==0.10.32 'git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a'
if ($LASTEXITCODE -ne 0) { throw '推論依存の導入に失敗しました。' }
& $ComputePython -c 'import torch,mediapipe; from moge.model.v3 import MoGeModel; print(torch.__version__, torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw '推論モジュールを読み込めません。ログを確認してください。' }
Write-Host "制作画面の推論環境Pythonに設定してください: $ComputePython"
