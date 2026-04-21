$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $root "data"
$sourceDir = Join-Path $dataDir "Bangla-Bayanno-Full"
$repoUrl = "https://huggingface.co/datasets/Remian9080/Bangla-Bayanno-Full"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed. Install git and retry."
}

if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    throw "git-lfs is not installed. Install git-lfs and retry."
}

New-Item -ItemType Directory -Path $dataDir -Force | Out-Null

git lfs install | Out-Host

if (-not (Test-Path $sourceDir)) {
    git clone $repoUrl $sourceDir | Out-Host
}

Copy-Item (Join-Path $sourceDir "qa.json") (Join-Path $dataDir "qa.json") -Force
New-Item -ItemType Directory -Path (Join-Path $dataDir "images") -Force | Out-Null
Copy-Item (Join-Path $sourceDir "images\*") (Join-Path $dataDir "images") -Recurse -Force

Write-Host "Dataset prepared in $dataDir"
