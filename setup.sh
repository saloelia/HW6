#!/bin/bash
# Setup script for Prompt Engineering Experiment

set -e

echo "================================================"
echo "Prompt Engineering Experiment - Setup"
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo ""
echo "Installing package in development mode..."
pip install -e .

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp config/.env.example .env
    echo "IMPORTANT: Edit .env and add your OPENAI_API_KEY"
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the experiment:"
echo "  prompt-experiment --dataset data/raw/dataset.json"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To run the Jupyter notebook:"
echo "  jupyter notebook notebooks/analysis.ipynb"
echo ""
