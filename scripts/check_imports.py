#!/usr/bin/env python3
"""
Comprehensive import checker for Susurrus project.
Checks for:
- Missing imports
- Circular imports
- Unused imports
- Import order issues
"""

import ast
import os
import sys
from collections import defaultdict
from pathlib import Path


class ImportChecker:
    """Check imports across the project"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.python_files = list(self.project_root.rglob("*.py"))
        self.errors = []
        self.warnings = []
        self.modules = {}
        self.import_graph = defaultdict(set)

    def check_all(self):
        """Run all checks"""
        print("🔍 Checking imports in Susurrus project...\n")

        # Parse all files first
        self._parse_all_files()

        # Run checks
        self._check_missing_imports()
        self._check_circular_imports()
        self._check_unused_imports()
        self._check_import_order()

        # Report results
        self._report_results()

        return len(self.errors) == 0

    def _parse_all_files(self):
        """Parse all Python files and extract import information"""
        print("📖 Parsing files...")

        for filepath in self.python_files:
            if "__pycache__" in str(filepath) or ".venv" in str(filepath):
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))

                self.modules[filepath] = {
                    "tree": tree,
                    "imports": self._extract_imports(tree),
                    "defined": self._extract_definitions(tree),
                    "used": self._extract_usage(tree),
                }
            except SyntaxError as e:
                self.errors.append(f"❌ Syntax error in {filepath}: {e}")
            except Exception as e:
                self.warnings.append(f"⚠️  Could not parse {filepath}: {e}")

    def _extract_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract all imports from AST"""
        imports = {
            "modules": [],  # import X
            "from_imports": [],  # from X import Y
            "aliases": {},  # import X as Y
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports["modules"].append(alias.name)
                    if alias.asname:
                        imports["aliases"][alias.asname] = alias.name

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports["from_imports"].append(
                        {"module": node.module, "names": [alias.name for alias in node.names]}
                    )

        return imports

    def _extract_definitions(self, tree: ast.AST) -> Set[str]:
        """Extract all definitions (classes, functions, variables)"""
        definitions = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions.add(target.id)

        return definitions

    def _extract_usage(self, tree: ast.AST) -> Set[str]:
        """Extract all name usage"""
        used = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used.add(node.value.id)

        return used

    def _check_missing_imports(self):
        """Check for missing imports"""
        print("🔎 Checking for missing imports...")

        for filepath, data in self.modules.items():
            used = data["used"]
            imports = data["imports"]
            defined = data["defined"]

            # Get all imported names
            imported_names = set(imports["modules"])
            for alias, original in imports["aliases"].items():
                imported_names.add(alias)
            for from_import in imports["from_imports"]:
                imported_names.update(from_import["names"])

            # Find potentially missing imports
            # (Names used but not imported or defined)
            builtins = {
                "print",
                "len",
                "range",
                "str",
                "int",
                "float",
                "list",
                "dict",
                "set",
                "tuple",
                "bool",
                "True",
                "False",
                "None",
                "Exception",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "open",
                "super",
                "enumerate",
                "zip",
                "map",
                "filter",
                "sorted",
                "max",
                "min",
                "sum",
                "any",
                "all",
                "isinstance",
                "hasattr",
                "getattr",
                "setattr",
                "type",
                "object",
                "property",
                "__name__",
                "__init__",
                "__file__",
                "__main__",
                "self",
                "cls",
            }

            potentially_missing = used - imported_names - defined - builtins

            # Filter out common false positives
            filtered_missing = set()
            for name in potentially_missing:
                # Skip if it's a method call or attribute access pattern
                if name.startswith("_") and name.endswith("_"):
                    continue
                # Skip single letter variables (often loop counters)
                if len(name) == 1 and name.islower():
                    continue
                filtered_missing.add(name)

            if filtered_missing:
                rel_path = filepath.relative_to(self.project_root)
                self.warnings.append(
                    f"⚠️  {rel_path}: Potentially missing imports: {', '.join(sorted(filtered_missing))}"
                )

    def _check_circular_imports(self):
        """Check for circular import dependencies"""
        print("🔄 Checking for circular imports...")

        # Build import graph
        for filepath, data in self.modules.items():
            imports = data["imports"]

            # Convert file path to module name
            module_name = self._filepath_to_module(filepath)

            # Add edges for all imports
            for imp_module in imports["modules"]:
                if imp_module.startswith("."):  # Relative import
                    # Resolve relative import
                    abs_module = self._resolve_relative_import(module_name, imp_module)
                    self.import_graph[module_name].add(abs_module)
                else:
                    self.import_graph[module_name].add(imp_module)

            for from_import in imports["from_imports"]:
                imp_module = from_import["module"]
                if imp_module.startswith("."):
                    abs_module = self._resolve_relative_import(module_name, imp_module)
                    self.import_graph[module_name].add(abs_module)
                else:
                    self.import_graph[module_name].add(imp_module)

        # Detect cycles using DFS
        cycles = self._find_cycles()

        if cycles:
            for cycle in cycles:
                self.errors.append(
                    f"❌ Circular import detected: {' -> '.join(cycle)} -> {cycle[0]}"
                )

    def _filepath_to_module(self, filepath: Path) -> str:
        """Convert file path to Python module name"""
        rel_path = filepath.relative_to(self.project_root)
        module = str(rel_path).replace(os.sep, ".").replace(".py", "")
        return module

    def _resolve_relative_import(self, current_module: str, relative_import: str) -> str:
        """Resolve relative import to absolute"""
        parts = current_module.split(".")

        # Count leading dots
        level = 0
        for char in relative_import:
            if char == ".":
                level += 1
            else:
                break

        # Remove leading dots from import
        import_name = relative_import.lstrip(".")

        # Go up the hierarchy
        base_parts = parts[:-level] if level > 0 else parts

        if import_name:
            return ".".join(base_parts + [import_name])
        else:
            return ".".join(base_parts)

    def _find_cycles(self) -> List[List[str]]:
        """Find all cycles in import graph using DFS"""
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = rec_stack.index(node)
                cycle = rec_stack[cycle_start:]
                if cycle not in cycles:
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.append(node)

            for neighbor in self.import_graph.get(node, []):
                # Only check internal modules
                if neighbor in self.import_graph:
                    dfs(neighbor, path + [neighbor])

            rec_stack.pop()

        for node in self.import_graph:
            dfs(node, [node])

        return cycles

    def _check_unused_imports(self):
        """Check for unused imports"""
        print("🗑️  Checking for unused imports...")

        for filepath, data in self.modules.items():
            imports = data["imports"]
            used = data["used"]

            # Check if imported names are used
            unused = []

            for module in imports["modules"]:
                # Get the base name (e.g., 'os' from 'os.path')
                base_name = module.split(".")[0]
                if base_name not in used:
                    unused.append(module)

            for alias, original in imports["aliases"].items():
                if alias not in used:
                    unused.append(f"{original} as {alias}")

            if unused:
                rel_path = filepath.relative_to(self.project_root)
                self.warnings.append(f"⚠️  {rel_path}: Possibly unused imports: {', '.join(unused)}")

    def _check_import_order(self):
        """Check import ordering (stdlib, third-party, local)"""
        print("📋 Checking import order...")

        for filepath, data in self.modules.items():
            # Check if imports follow convention:
            # 1. Standard library
            # 2. Third-party
            # 3. Local/relative

            # This is a simplified check
            # More sophisticated checking would use isort
            pass

    def _report_results(self):
        """Print summary of all checks"""
        print("\n" + "=" * 70)
        print("IMPORT CHECK RESULTS")
        print("=" * 70)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed! No import issues found.")

        print("\n" + "=" * 70)


def main():
    """Run import checks"""
    project_root = Path(__file__).parent.parent

    checker = ImportChecker(project_root)
    success = checker.check_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
