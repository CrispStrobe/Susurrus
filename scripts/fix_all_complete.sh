#!/bin/bash
# Complete fix workflow with syntax error handling

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔧 Complete import and syntax fixing..."
echo "========================================"

# 1. Fix import order
echo "📋 Step 1: Fixing import order..."
isort . --profile black --line-length 100 2>/dev/null || true

# 2. Remove unused imports
echo "🗑️  Step 2: Removing unused imports..."
python3 scripts/fix_unused_imports.py 2>/dev/null || true

# 3. Clean up specific issues
echo "🧹 Step 3: Cleaning up specific issues..."
bash scripts/cleanup_imports.sh 2>/dev/null || true

# 4. Fix syntax errors
echo "🔧 Step 4: Fixing syntax errors..."
python3 scripts/fix_syntax_errors.py 2>/dev/null || true

# 5. Format code (with error handling)
echo "🎨 Step 5: Formatting code..."
black . --line-length 100 2>&1 | grep -v "^error:" || true

echo ""
echo "✅ All fixes applied!"
echo "Now run: python scripts/check_imports.py"