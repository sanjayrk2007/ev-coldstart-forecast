#!/bin/bash
# Run this once after cloning the repo to set up your dev environment

set -e

echo "==> Creating virtual environment..."
python3.11 -m venv venv

echo "==> Activating venv..."
source venv/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Installing pre-commit hooks..."
pre-commit install

echo "==> Copying .env..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  Created .env from .env.example — fill in your values."
else
  echo "  .env already exists, skipping."
fi

echo ""
echo "✅ Dev environment ready."
echo ""
echo "To start MLflow UI:"
echo "  source venv/bin/activate && mlflow ui"
echo "  Then open http://localhost:5000"
