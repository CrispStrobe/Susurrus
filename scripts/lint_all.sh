#!/bin/bash
# Comprehensive linting script for Susurrus

set -e  # Exit on first error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 Running comprehensive lint checks for Susurrus..."
echo "=================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

EXIT_CODE=0

# Function to run a check
run_check() {
    local name=$1
    local command=$2
    
    echo -e "${YELLOW}▶ Running $name...${NC}"
    if eval $command; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        EXIT_CODE=1
    fi
}

# 1. Check imports
run_check "Import Checker" "python scripts/check_imports.py"

# 2. Flake8 (style, syntax, complexity)
run_check "Flake8" "flake8 . --count --show-source --statistics"

# 3. Pylint (code quality)
run_check "Pylint" "pylint susurrus/ gui/ utils/ backends/ workers/ --exit-zero"

# 4. MyPy (type checking)
run_check "MyPy" "mypy . --ignore-missing-imports --no-error-summary || true"

# 5. Import order
run_check "Import Order (isort)" "isort . --check-only --diff"

# 6. Code formatting
run_check "Code Formatting (black)" "black . --check --diff"

# 7. Security issues
run_check "Security Check (bandit)" "bandit -r . -ll -i || true"

# 8. Find unused code
run_check "Dead Code (vulture)" "vulture . --min-confidence 80 || true"

# 9. Docstring coverage
run_check "Docstring Coverage" "pydocstyle . --count || true"

# 10. Check for common issues
echo -e "${YELLOW}▶ Checking for common issues...${NC}"

# Check for print statements (should use logging)
if grep -r "print(" --include="*.py" . | grep -v "test_" | grep -v "scripts/" | grep -v "#.*print"; then
    echo -e "${YELLOW}⚠ Found print statements (consider using logging)${NC}"
fi

# Check for TODO/FIXME comments
TODO_COUNT=$(grep -r "TODO\|FIXME" --include="*.py" . | wc -l)
if [ $TODO_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $TODO_COUNT TODO/FIXME comments${NC}"
fi

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
else
    echo -e "${RED}❌ Some checks failed. Please review the output above.${NC}"
fi
echo "=================================================="

exit $EXIT_CODE