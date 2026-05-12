#!/bin/bash

echo "Checking Python version..."
python_version=$(python --version 2>&1)

if [[ $python_version == *"3.10"* ]]; then
    echo "✓ Python 3.10.x"
else
    echo "✗ Python 3.10 required"
    exit 1
fi

echo "Checking required packages..."

packages=("numpy" "pandas" "matplotlib" "scikit-learn" "torch")

for pkg in "${packages[@]}"; do
    python -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ $pkg installed"
    else
        echo "✗ $pkg NOT installed"
        exit 1
    fi
done

echo "Checking dataset..."
if [ -d "data/raw" ]; then
    echo "✓ data/raw directory exists"
else
    echo "✗ data/raw missing"
    exit 1
fi

echo ""
echo "Environment check PASSED"
