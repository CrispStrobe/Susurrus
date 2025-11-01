#!/usr/bin/env python3
"""
Automatically remove unused imports from Python files.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set


class UnusedImportRemover:
    """Remove unused imports from files"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.files_modified = 0
        self.imports_removed = 0

    def process_all_files(self):
        """Process all Python files"""
        python_files = list(self.project_root.rglob("*.py"))

        for filepath in python_files:
            if any(x in str(filepath) for x in ["__pycache__", ".venv", "venv", ".git"]):
                continue

            if "__init__.py" in str(filepath):
                # Skip __init__.py files - they often have intentional unused imports
                continue

            self.process_file(filepath)

        print(f"\n✅ Modified {self.files_modified} files")
        print(f"✅ Removed {self.imports_removed} unused imports")

    def process_file(self, filepath: Path):
        """Process a single file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original_content = f.read()
                tree = ast.parse(original_content)

            # Get imports and usage
            imports = self._get_imports(tree)
            used_names = self._get_used_names(tree)

            # Find unused imports
            unused = self._find_unused(imports, used_names, original_content)

            if unused:
                # Remove unused imports
                new_content = self._remove_imports(original_content, unused)

                if new_content != original_content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)

                    print(
                        f"✓ {filepath.relative_to(self.project_root)}: "
                        f"removed {len(unused)} imports"
                    )
                    self.files_modified += 1
                    self.imports_removed += len(unused)

        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")

    def _get_imports(self, tree: ast.AST) -> List[Dict]:
        """Get all import statements with line numbers"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "lineno": node.lineno,
                            "type": "import",
                            "module": alias.name,
                            "name": alias.asname or alias.name.split(".")[0],
                            "full_line": node,
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        if alias.name != "*":
                            imports.append(
                                {
                                    "lineno": node.lineno,
                                    "type": "from",
                                    "module": node.module,
                                    "name": alias.asname or alias.name,
                                    "imported": alias.name,
                                    "full_line": node,
                                }
                            )

        return imports

    def _get_used_names(self, tree: ast.AST) -> Set[str]:
        """Get all used names, excluding import statements"""
        used = set()

        class UsageVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                used.add(node.id)

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    used.add(node.value.id)
                self.generic_visit(node)

            def visit_Import(self, node):
                pass  # Don't visit import statements

            def visit_ImportFrom(self, node):
                pass  # Don't visit import statements

        visitor = UsageVisitor()
        visitor.visit(tree)

        return used

    def _find_unused(self, imports: List[Dict], used: Set[str], content: str) -> List[Dict]:
        """Find unused imports"""
        unused = []

        for imp in imports:
            name = imp["name"]

            # Check if used
            if name not in used:
                # Also check if it's used in type annotations (as string)
                if f"'{name}'" not in content and f'"{name}"' not in content:
                    # Check if it's logging (often imported but used differently)
                    if name == "logging" and "logging." in content:
                        continue

                    unused.append(imp)

        return unused

    def _remove_imports(self, content: str, unused: List[Dict]) -> str:
        """Remove unused import lines"""
        lines = content.split("\n")
        lines_to_remove = set(imp["lineno"] - 1 for imp in unused)

        new_lines = []
        for i, line in enumerate(lines):
            if i not in lines_to_remove:
                new_lines.append(line)

        return "\n".join(new_lines)


def main():
    project_root = Path(__file__).parent.parent

    print("🔧 Removing unused imports...")
    remover = UnusedImportRemover(project_root)
    remover.process_all_files()


if __name__ == "__main__":
    main()
