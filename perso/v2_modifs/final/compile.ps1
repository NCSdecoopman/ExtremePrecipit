$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$miktexBin = Join-Path $env:USERPROFILE 'scoop\apps\miktex\current\texmfs\install\miktex\bin\x64'
if (-not (Test-Path (Join-Path $miktexBin 'pdflatex.exe'))) {
    Write-Error "MiKTeX introuvable. Installez-le avec: scoop install miktex"
}

$env:Path = "$miktexBin;$env:Path"
$env:MIKTEX_ALLOW_UNSAFE_ADMIN_INSTALL = '1'

initexmf --set-config-value '[MPM]AutoInstall=1' 2>$null

$args = @('-interaction=nonstopmode', '-halt-on-error', 'main.tex')
foreach ($i in 1..2) { & pdflatex @args }
& bibtex main
foreach ($i in 1..2) { & pdflatex @args }

Write-Host "OK: main.pdf ($((Get-Item main.pdf).Length) octets)"
