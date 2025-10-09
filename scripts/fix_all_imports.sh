#!/bin/bash
# Comprehensive import fixing

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔧 Comprehensive import fixing..."
echo "================================="

# 1. Fix import order
echo "📋 Step 1: Fixing import order with isort..."
isort . --profile black --line-length 100

# 2. Remove unused imports automatically
echo "🗑️  Step 2: Removing unused imports..."
python3 scripts/fix_unused_imports.py

# 3. Clean up specific known issues
echo "🧹 Step 3: Cleaning up specific issues..."
bash scripts/cleanup_imports.sh

# 4. Format code
echo "🎨 Step 4: Formatting code with black..."
black . --line-length 100

echo ""
echo "✅ All fixes applied!"
echo "Run './scripts/lint_all.sh' to verify results"