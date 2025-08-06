#!/usr/bin/env bash
# exit on error
set -o errexit

# Make sure the script is executable
chmod +x build.sh

# Install system dependencies for OCR
echo "--- Installing Tesseract OCR ---"
apt-get update && apt-get install -y tesseract-ocr

# Verify Tesseract installation
echo "--- Verifying Tesseract installation ---"
tesseract --version

# Install Python dependencies
echo "--- Installing Python libraries ---"
pip install -r requirements.txt
