#!/usr/bin/env bash
set -e

echo "=========================================="
echo "    Onco_Bot Environment Setup Script     "
echo "=========================================="
echo ""

# 1. Detect Operating System
OS="$(uname -s)"
echo "Detected Operating System: $OS"

if [ "$OS" = "Darwin" ]; then
    echo "macOS detected. Installing system dependencies via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install Homebrew first (https://brew.sh/)."
        exit 1
    fi
    brew install openslide
elif [ "$OS" = "Linux" ]; then
    echo "Linux detected. Installing system dependencies via APT..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y openslide-tools libopenslide0
    else
        echo "Error: APT package manager not found. Please install 'openslide' manually using your distribution's package manager."
        exit 1
    fi
else
    echo "Warning: Unsupported OS ($OS). Please ensure OpenSlide C-libraries are installed manually."
fi

echo ""
echo "=========================================="
echo "    Python Environment Configuration      "
echo "=========================================="

# 2. Check for Conda vs Venv
if command -v conda &> /dev/null; then
    echo "Conda detected! Setting up Conda environment 'onco_bot'..."
    # Check if env exists
    if conda env list | grep -q 'onco_bot '; then
        echo "Conda environment 'onco_bot' already exists. Updating..."
    else
        conda create -n onco_bot python=3.10 -y
    fi
    echo "Installing Python dependencies into Conda..."
    # We use conda run to install without needing the user to activate the shell first
    conda run -n onco_bot pip install -r requirements.txt
    
    echo ""
    echo "✅ Setup Complete!"
    echo "To run the sample inference script, execute:"
    echo "  conda activate onco_bot"
    echo "  python run_sample.py"
    
else
    echo "Conda not found. Falling back to Python virtual environment (venv)..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    echo "Activating virtual environment and installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo ""
    echo "✅ Setup Complete!"
    echo "To run the sample inference script, execute:"
    echo "  source venv/bin/activate"
    echo "  python run_sample.py"
fi
