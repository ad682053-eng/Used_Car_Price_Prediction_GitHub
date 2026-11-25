#!/bin/zsh
set -e
cd "$(dirname "$0")"

PROJECT_PYTHON=""
if command -v python3.11 >/dev/null 2>&1; then
  PROJECT_PYTHON="python3.11"
else
  echo "Python 3.11 was not found. Install Python 3.11 for macOS, then run this file again."
  echo "Download: https://www.python.org/downloads/macos/"
  read -r "?Press Return to close..."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating project virtual environment..."
  "$PROJECT_PYTHON" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f "model/used_car_price_model.cbm" ]; then
  echo "Saved model not found. Training from data/used_cars.csv..."
  python train_model.py
fi

echo "Starting Used Car Price Prediction..."
echo "Keep this Terminal window open while using the app."
python -m streamlit run app.py
