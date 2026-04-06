#!/bin/bash
# Setup script for Phase 1

echo "Setting up Session-Adaptive News Ranker - Phase 1"
echo "=================================================="

# Create virtual environment
echo -e "\n[1/3] Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo -e "\n[2/3] Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies
echo -e "\n[3/3] Installing dependencies..."
pip install -r requirements.txt

echo -e "\n=================================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download MIND dataset from: https://msnews.github.io/"
echo "2. Place behaviors.tsv and news.tsv in data/raw/"
echo "3. Run: python pipeline.py"
echo ""
echo "To activate environment later:"
echo "  Linux/Mac: source venv/bin/activate"
echo "  Windows: venv\\Scripts\\activate"
