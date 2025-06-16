#!/bin/bash

n=".venv"

set -e

if [ ! -d "$n" ]; then
  echo "Virtual environment not found..."
else
  echo "Creating virtual environment..."
  python3 -m venv "$n"
fi

echo "Activating virtual environment..."
source "./$n/bin/activate"

echo "Installing necessary Python libraries..."
pip install codecarbon dash notebook lightgbm matplotlib numpy pandas scikit-learn

echo "Setup done. To run the application type './run.sh'"
