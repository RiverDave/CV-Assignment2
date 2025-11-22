#!/bin/bash

# Quick Python environment setup script
# Usage: bash setup_env.sh [env_name]

ENV_NAME="${1:-.venv}"

echo "Creating Python virtual environment: $ENV_NAME"
python3 -m venv "$ENV_NAME"

echo "Activating environment..."
source "$ENV_NAME/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "✓ Environment ready!"
echo "Environment name: $ENV_NAME"
echo "To activate in the future, run: source $ENV_NAME/bin/activate"
echo "To deactivate, run: deactivate"
