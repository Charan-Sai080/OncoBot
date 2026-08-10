Write-Host "=========================================="
Write-Host "    Onco_Bot Windows Setup Script         "
Write-Host "=========================================="

$openslideUrl = "https://github.com/openslide/openslide-bin/releases/download/v4.0.0.3/openslide-win64-20231122.zip"
$zipPath = "openslide-win64.zip"
$extractPath = "openslide-bin"

if (-Not (Test-Path -Path $extractPath)) {
    Write-Host "Downloading Pre-compiled OpenSlide C-Libraries for Windows..."
    Invoke-WebRequest -Uri $openslideUrl -OutFile $zipPath
    Write-Host "Extracting OpenSlide..."
    Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
    Remove-Item -Path $zipPath
    Write-Host "OpenSlide binaries downloaded successfully!"
} else {
    Write-Host "OpenSlide binaries already exist."
}

if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "Conda detected. Setting up onco_bot environment..."
    conda create -n onco_bot python=3.10 -y
    conda run -n onco_bot pip install -r requirements.txt
    Write-Host ""
    Write-Host "Setup Complete!"
    Write-Host "To run: conda activate onco_bot; python run_sample.py"
} else {
    Write-Host "Conda not found. Falling back to Python venv..."
    if (-Not (Test-Path -Path "venv")) {
        python -m venv venv
    }
    .\venv\Scripts\pip install -r requirements.txt
    Write-Host ""
    Write-Host "Setup Complete!"
    Write-Host "To run: .\venv\Scripts\activate; python run_sample.py"
}
