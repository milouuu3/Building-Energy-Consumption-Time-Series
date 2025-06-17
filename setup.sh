#!/bin/bash

n=".venv"

set -e

if [ ! -d "$n" ]; then
  echo "Virtual environment not found..."
  python3 -m venv "$n"
else
  echo "Creating virtual environment..."
fi

echo "Activating virtual environment..."
source "./$n/bin/activate"

echo "Installing necessary Python libraries..."
pip install codecarbon dash notebook lightgbm matplotlib numpy pandas scikit-learn

echo "Setup done. To run the application type './run.sh'"
