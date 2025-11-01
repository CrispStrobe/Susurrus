#!/bin/bash
# Auto-fix what we can

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔧 Auto-fixing code issues..."

# 1. Fix import order
echo "📋 Fixing import order..."
isort .

# 2. Fix code formatting
echo "🎨 Fixing code formatting..."
black .

# 3. Remove unused imports (if autopep8 installed)
if command -v autopep8 &> /dev/null; then
    echo "🗑️  Removing unused imports..."
    autopep8 --in-place --recursive --select=W6 .
fi

echo "✅ Auto-fix complete! Run ./scripts/lint_all.sh to check results."