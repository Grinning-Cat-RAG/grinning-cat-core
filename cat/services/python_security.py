import ast
from pathlib import Path


FORBIDDEN_MODULES = {
    "ctypes", "cffi", "pty", "termios",
    "socket",  # direct networking
    "pickle", "shelve",  # arbitrary deserialization
    "zipimport", "pkgutil",
}

# `importlib` is deliberately NOT banned wholesale: `importlib.util.find_spec` is a
# pure availability check that never executes the module, and with `__import__` and
# `pkgutil` already forbidden a plugin has no other way to ask "is this package
# installed?". Only the entry points that can actually load code are blocked.
FORBIDDEN_IMPORTLIB_NAMES = {
    "import_module", "reload", "__import__",
    "module_from_spec", "spec_from_file_location", "spec_from_loader",
    "SourceFileLoader", "SourcelessFileLoader", "ExtensionFileLoader",
}

FORBIDDEN_BUILTINS = {
    "__import__", "exec", "eval", "compile",
    "open",  # use pathlib or a safe wrapper, instead
    "breakpoint",  # access to the debugger
    "globals", "locals", "vars",  # access to the environment
    "memoryview",  # access to raw memory
}

# Dunders that expose no object graph and no execution path: the ordinary protocol
# methods plus read-only metadata. Everything else stays forbidden — in particular
# __class__, __bases__, __subclasses__, __mro__, __globals__, __code__, __dict__,
# __builtins__, __getattribute__, __setattr__ and __reduce__.
DUNDER_WHITELIST = {
    "__init__", "__str__", "__repr__", "__len__", "__iter__", "__next__", "__enter__", "__exit__",
    "__module__", "__qualname__", "__name__", "__file__", "__doc__", "__version__", "__all__",
}


class ASTSecurityVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._depth = 0
        self._importlib_aliases = {"importlib"}

    def _fail(self, node: ast.AST, reason: str):
        raise SecurityError(f"{self.filepath}:{node.lineno}: {reason}")

    def generic_visit(self, node: ast.AST):
        self._depth += 1
        if self._depth > 200:
            raise SecurityError(f"{self.filepath}: AST too complex")
        super().generic_visit(node)
        self._depth -= 1

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._fail(node, f"import forbidden: {alias.name}")
            if root == "importlib" and alias.asname and alias.name == "importlib":
                self._importlib_aliases.add(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._fail(node, f"import forbidden: {node.module}")
            if root == "importlib":
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORTLIB_NAMES:
                        self._fail(node, f"dynamic import forbidden: {node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Case: direct name (exec, eval, open, ...)
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                self._fail(node, f"builtin forbidden: {node.func.id}")

        # Case: attribute (os.system, subprocess.Popen, ...)
        if isinstance(node.func, ast.Attribute):
            full = _reconstruct_attr(node.func)
            if full:
                # Block __dunder__ on whaetever object
                if any(
                    part.startswith("__") and part not in DUNDER_WHITELIST
                    for part in full.split(".")
                ):
                    self._fail(node, f"dunder access forbidden: {full}")

        # # Block getattr(x, "something") — classic bypass mean
        # if isinstance(node.func, ast.Name) and node.func.id == "getattr":
        #     self._fail(node, "getattr() forbidden (mean for bypass)")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Block the access to __class__, __bases__, __subclasses__, etc.
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in DUNDER_WHITELIST:
                self._fail(node, f"dunder access forbidden: {node.attr}")

        # importlib: the introspection surface is allowed, the loaders are not.
        full = _reconstruct_attr(node)
        if full:
            parts = full.split(".")
            if parts[0] in self._importlib_aliases and any(
                part in FORBIDDEN_IMPORTLIB_NAMES for part in parts[1:]
            ):
                self._fail(node, f"dynamic import forbidden: {full}")

        self.generic_visit(node)


def _reconstruct_attr(node: ast.Attribute) -> str | None:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


class SecurityError(Exception):
    pass


def ast_scan(filepath: Path) -> None:
    """Raise SecurityError if the file contains forbidden patterns."""
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as exc:
        raise SecurityError(f"Syntax error in {filepath}: {exc}") from exc
    ASTSecurityVisitor(str(filepath)).visit(tree)
