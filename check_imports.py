#!/usr/bin/env python3
"""
Improved import checker with fewer false positives.
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple  

class ImprovedImportChecker:
    """Check imports with better false positive filtering"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.python_files = list(self.project_root.rglob("*.py"))
        self.errors = []
        self.warnings = []
        self.modules = {}
        self.import_graph = defaultdict(set)

        # Comprehensive list of Python builtins and common names
        self.builtins = {
            # Built-in functions
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
            "AttributeError",
            "ImportError",
            "FileNotFoundError",
            "RuntimeError",
            "OSError",
            "NameError",
            "SyntaxError",
            "WindowsError",
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
            "delattr",
            "type",
            "object",
            "property",
            "staticmethod",
            "classmethod",
            "abs",
            "round",
            "pow",
            "divmod",
            "iter",
            "next",
            "reversed",
            "slice",
            "format",
            "chr",
            "ord",
            # Special names
            "__name__",
            "__init__",
            "__file__",
            "__main__",
            "__doc__",
            "__dict__",
            "__class__",
            "__bases__",
            "__module__",
            "self",
            "cls",
            # Common exception names (even if not imported)
            "e",
            "ex",
            "err",
            "error",
            "exception",
            # Common variable names (loop counters, etc)
            "i",
            "j",
            "k",
            "x",
            "y",
            "z",
            "n",
            "m",
            "a",
            "b",
            "c",
            "idx",
            "index",
            "count",
            "total",
            "value",
            "item",
            "key",
            "name",
            "path",
            "file",
            "filename",
            "data",
            "result",
            "args",
            "kwargs",
            "options",
            "params",
            "config",
            "settings",
            # Common iteration variables
            "line",
            "row",
            "col",
            "node",
            "child",
            "parent",
            "root",
            "chunk",
            "segment",
            "part",
            "piece",
            "token",
            # Common descriptive names
            "msg",
            "message",
            "text",
            "content",
            "output",
            "input",
            "source",
            "dest",
            "src",
            "dst",
            "url",
            "uri",
            "link",
            # Type names (often used in annotations or checks)
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Optional",
            "Union",
            "Any",
        }

    def check_all(self):
        """Run all checks"""
        print("🔍 Checking imports in Susurrus project...\n")

        self._parse_all_files()
        self._check_circular_imports()
        self._check_unused_imports()
        self._report_results()

        return len(self.errors) == 0

    def _parse_all_files(self):
        """Parse all Python files"""
        print("📖 Parsing files...")

        for filepath in self.python_files:
            if any(x in str(filepath) for x in ["__pycache__", ".venv", "venv", ".git"]):
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))

                self.modules[filepath] = {
                    "tree": tree,
                    "imports": self._extract_imports(tree),
                    "defined": self._extract_definitions(tree),
                    "used": self._extract_usage(tree),
                    "params": self._extract_parameters(tree),
                }
            except (SyntaxError, UnicodeDecodeError) as e:
                self.errors.append(f"❌ Cannot parse {filepath}: {e}")
            except Exception as e:
                self.warnings.append(f"⚠️  Parse warning for {filepath}: {e}")

    def _extract_imports(self, tree: ast.AST) -> Dict:
        """Extract all imports"""
        imports = {"modules": set(), "from_modules": set(), "names": set(), "aliases": {}}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports["modules"].add(alias.name)
                    base_name = alias.name.split(".")[0]
                    imports["names"].add(base_name)
                    if alias.asname:
                        imports["aliases"][alias.asname] = alias.name
                        imports["names"].add(alias.asname)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports["from_modules"].add(node.module)
                    for alias in node.names:
                        if alias.name != "*":
                            imports["names"].add(alias.name)
                            if alias.asname:
                                imports["aliases"][alias.asname] = alias.name
                                imports["names"].add(alias.asname)

        return imports

    def _extract_definitions(self, tree: ast.AST) -> Set[str]:
        """Extract all definitions"""
        definitions = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions.add(target.id)

        return definitions

    def _extract_parameters(self, tree: ast.AST) -> Set[str]:
        """Extract all function/method parameters"""
        params = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Regular arguments
                for arg in node.args.args:
                    params.add(arg.arg)
                # *args
                if node.args.vararg:
                    params.add(node.args.vararg.arg)
                # **kwargs
                if node.args.kwarg:
                    params.add(node.args.kwarg.arg)
                # Keyword-only arguments
                for arg in node.args.kwonlyargs:
                    params.add(arg.arg)

            elif isinstance(node, (ast.ExceptHandler,)):
                # Exception handler variables
                if node.name:
                    params.add(node.name)

            elif isinstance(node, (ast.For, ast.comprehension)):
                # Loop variables
                if isinstance(node, ast.For):
                    target = node.target
                else:
                    target = node.target

                if isinstance(target, ast.Name):
                    params.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            params.add(elt.id)

        return params

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

    def _check_circular_imports(self):
        """Check for circular import dependencies"""
        print("🔄 Checking for circular imports...")

        # Build import graph
        for filepath, data in self.modules.items():
            imports = data["imports"]
            module_name = self._filepath_to_module(filepath)

            # Add all imports to graph
            for imp_module in imports["modules"] | imports["from_modules"]:
                if not imp_module.startswith("."):  # Skip relative imports for now
                    self.import_graph[module_name].add(imp_module)

        # Detect cycles
        cycles = self._find_cycles()

        if cycles:
            for cycle in cycles:
                if len(cycle) > 1:  # Only report real cycles
                    self.errors.append(f"❌ Circular import: {' -> '.join(cycle)}")

    def _filepath_to_module(self, filepath: Path) -> str:
        """Convert file path to module name"""
        try:
            rel_path = filepath.relative_to(self.project_root)
            module = str(rel_path).replace(os.sep, ".").replace(".py", "")
            if module.endswith(".__init__"):
                module = module[:-9]
            return module
        except ValueError:
            return str(filepath)

    def _find_cycles(self) -> List[List[str]]:
        """Find cycles in import graph"""
        cycles = []
        visited = set()

        def dfs(node, path, rec_stack):
            if node in rec_stack:
                cycle_start = rec_stack.index(node)
                cycle = rec_stack[cycle_start:]
                if len(cycle) > 1 and cycle not in cycles:
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.append(node)

            for neighbor in self.import_graph.get(node, []):
                if neighbor in self.import_graph:
                    dfs(neighbor, path + [neighbor], rec_stack[:])

            rec_stack.pop()

        for node in self.import_graph:
            dfs(node, [node], [])

        return cycles

    def _check_unused_imports(self):
        """Check for unused imports"""
        print("🗑️  Checking for unused imports...")

        for filepath, data in self.modules.items():
            imports = data["imports"]
            used = data["used"]

            unused = []

            # Check module imports
            for module in imports["modules"]:
                base_name = module.split(".")[0]
                if base_name not in used:
                    # Check if it's used in annotations
                    file_content = open(filepath).read()
                    if (
                        f"'{base_name}'" not in file_content
                        and f'"{base_name}"' not in file_content
                    ):
                        unused.append(module)

            # Check aliases
            for alias, original in imports["aliases"].items():
                if alias not in used:
                    unused.append(f"{original} as {alias}")

            if unused:
                rel_path = filepath.relative_to(self.project_root)
                # Only warn if it's not an __init__.py (those often have unused imports)
                if "__init__.py" not in str(filepath):
                    self.warnings.append(f"⚠️  {rel_path}: Unused imports: {', '.join(unused)}")

    def _report_results(self):
        """Print summary"""
        print("\n" + "=" * 70)
        print("IMPORT CHECK RESULTS")
        print("=" * 70)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:20]:  # Limit output
                print(f"  {warning}")
            if len(self.warnings) > 20:
                print(f"  ... and {len(self.warnings) - 20} more warnings")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")

        print("\n" + "=" * 70)


def main():
    project_root = Path(__file__).parent.parent
    checker = ImprovedImportChecker(project_root)
    success = checker.check_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
