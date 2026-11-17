param(
    [switch]$SkipTests,
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

$localBuildTools = Join-Path $projectRoot ".buildtools"
if (Test-Path -LiteralPath $localBuildTools) {
    $env:PYTHONPATH = "$localBuildTools;$env:PYTHONPATH"
}

python -c "import numpy, scipy, matplotlib, PySide6; print('Runtime dependencies OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Runtime dependencies are missing. Install requirements.txt on the build PC."
}

python -c "import PyInstaller; print('PyInstaller', PyInstaller.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is missing. Run: python -m pip install 'pyinstaller>=6,<7'"
}

if (-not $SkipTests) {
    New-Item -ItemType Directory -Force (Join-Path $projectRoot ".tmp") | Out-Null
    $env:QT_QPA_PLATFORM = "offscreen"
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Tests failed; portable build aborted."
    }
}

python -m PyInstaller --noconfirm --clean BackgroundCorrectionStudio.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

# The smoke test intentionally removes Miniforge/Conda from PATH.  This catches
# missing native Qt DLLs that would otherwise be masked on the build PC.
$portableExe = Join-Path $projectRoot "dist\BackgroundCorrectionStudio\BackgroundCorrectionStudio.exe"
$savedPath = $env:PATH
$savedQtPlatform = $env:QT_QPA_PLATFORM
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    $env:QT_QPA_PLATFORM = "offscreen"
    $smoke = Start-Process -FilePath $portableExe -ArgumentList "--smoke-test" -WindowStyle Hidden -Wait -PassThru
    if ($smoke.ExitCode -ne 0) {
        throw "Portable executable smoke test failed with exit code $($smoke.ExitCode)."
    }
}
finally {
    $env:PATH = $savedPath
    $env:QT_QPA_PLATFORM = $savedQtPlatform
}

$archive = Join-Path $projectRoot "dist\BackgroundCorrectionStudio-portable.zip"
if (-not $SkipArchive) {
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    tar.exe -a -c -f $archive -C (Join-Path $projectRoot "dist") BackgroundCorrectionStudio
    if ($LASTEXITCODE -ne 0) {
        throw "Portable ZIP creation failed."
    }
}

$buildDirectory = Join-Path $projectRoot "build"
if (Test-Path -LiteralPath $buildDirectory) {
    Remove-Item -LiteralPath $buildDirectory -Recurse -Force
}
$testScratch = Join-Path $projectRoot ".tmp"
if ((Test-Path -LiteralPath $testScratch) -and -not (Get-ChildItem -LiteralPath $testScratch -Force)) {
    Remove-Item -LiteralPath $testScratch -Force
}

Write-Host "Portable application: dist\BackgroundCorrectionStudio\BackgroundCorrectionStudio.exe"
if (-not $SkipArchive) {
    Write-Host "Distribution archive: $archive"
} else {
    Write-Host "Distribution archive skipped."
}
