#!/bin/bash
# Clean up specific known issues

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧹 Cleaning up known import issues..."

# Remove unused logging imports from __init__.py files
find . -name "__init__.py" -type f | while read file; do
    if grep -q "^import logging$" "$file" && ! grep -q "logging\." "$file"; then
        echo "  Removing unused logging from $file"
        sed -i.bak '/^import logging$/d' "$file" && rm "${file}.bak"
    fi
done

# Remove unused imports from specific files
echo "  Cleaning config.py..."
python3 << 'EOF'
import re

with open('config.py', 'r') as f:
    content = f.read()

# Remove mlx import if not used
if 'import mlx' in content and 'mlx.' not in content.replace('import mlx', ''):
    content = re.sub(r'^import mlx\n', '', content, flags=re.MULTILINE)

with open('config.py', 'w') as f:
    f.write(content)
EOF

echo "  Cleaning utils/format_utils.py..."
python3 << 'EOF'
with open('utils/format_utils.py', 'r') as f:
    lines = f.readlines()

new_lines = [line for line in lines if not line.strip() == 'import logging']

with open('utils/format_utils.py', 'w') as f:
    f.writelines(new_lines)
EOF

echo "✅ Cleanup complete!"