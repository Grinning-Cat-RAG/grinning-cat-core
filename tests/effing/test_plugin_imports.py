"""Static guards on the plugin's own import graph.

Two defects this package has already hit, both of which only blow up when the
plugin is loaded by a running Cat (an ImportError at load time, or a NameError
on the first hook call) and neither of which any unit test would notice:

- a module importing the registry from the ``ingestion_status`` plugin, the
  one this package absorbed, which no longer exists;
- ``plugin.py`` calling helpers defined in a sibling module without importing
  them.

Pure ``ast`` inspection: it needs neither the ``cat`` package nor a running
Cat, so it stays a cheap regression guard on the package layout. The plugin
folder is located by walking up to the folder holding ``plugin.json``; when
the file is copied into the host's ``tests/`` (the EffING-vs-CAT harness) the
walk-up fails and the installed plugin folder is resolved via ``cat.utils``
(lazy import, so the in-repo run stays cat-free).
"""
import ast
import builtins
from pathlib import Path


def _plugin_dir() -> Path:
    """The plugin folder: walk up from this file to the folder holding plugin.json.

    Falls back to the installed plugin folder when the file runs from the
    host's ``tests/`` (copied harness), where no ``plugin.json`` is nearby.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / "plugin.json").is_file():
            return candidate
    from cat import utils  # lazy: keeps in-repo runs cat-free

    return Path(utils.__file__).resolve().parent / "core_plugins" / "cat_efficient_ingestion"


PLUGIN_DIR = _plugin_dir()
MODULES = {p.stem: p for p in PLUGIN_DIR.glob("*.py")}
BUILTINS = set(dir(builtins))


def _imports(path: Path):
    """Yield every ``(module, level, names)`` imported by a source file."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            yield node.module, node.level, [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0, []


def _exported_names(path: Path) -> set:
    """Top-level names a module defines or re-exports."""
    exported = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported.add(node.name)
        elif isinstance(node, ast.Assign):
            exported.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exported.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            exported.update(a.asname or a.name.split(".")[0] for a in node.names)
    return exported


def _bound_names(tree: ast.AST) -> set:
    """Every name bound anywhere in a module (assignment, def, import, arg...)."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def _loaded_names(tree: ast.AST) -> set:
    """Every name read anywhere in a module."""
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def test_no_import_from_the_former_ingestion_status_plugin():
    """No module imports from the absorbed ``ingestion_status`` plugin (it is gone).

    Matched on the package segment, so it holds wherever that plugin used to be
    installed (``cat.core_plugins.ingestion_status``, ``cat.plugins.*``, ...).
    """
    gone = {"ingestion_status", "cat_ingestion_status"}
    offenders = [
        f"{path.name}: {module}"
        for path in sorted(MODULES.values())
        for module, level, _ in _imports(path)
        if level == 0 and module and gone & set(module.split("."))
    ]
    assert offenders == [], f"stale imports of the removed plugin: {offenders}"


def test_intra_package_imports_are_relative_and_resolvable():
    """Sibling modules are imported relatively, and every target exists.

    An absolute import of a sibling hardcodes where the plugin is installed, so
    it breaks as soon as the package moves; it is flagged whatever the prefix.
    """
    missing = []
    absolute_self = []
    for path in sorted(MODULES.values()):
        for module, level, names in _imports(path):
            if level == 0:
                if not module or "." not in module:
                    continue
                sibling = module.rsplit(".", 1)[1]
                if sibling in MODULES and sibling != path.stem:
                    if set(names) <= _exported_names(MODULES[sibling]):
                        absolute_self.append(f"{path.name}: {module}")
                continue
            if module not in MODULES:
                missing.append(f"{path.name}: .{module}")
                continue
            exported = _exported_names(MODULES[module])
            missing.extend(
                f"{path.name}: .{module}.{name}" for name in names if name not in exported
            )
    assert absolute_self == [], f"intra-package imports must be relative: {absolute_self}"
    assert missing == [], f"unresolvable relative imports: {missing}"


def test_no_sibling_helper_used_without_importing_it():
    """A name defined in a sibling module and used here must be imported.

    Catches the NameError that only surfaces when the hook actually runs.
    """
    offenders = []
    trees = {name: ast.parse(path.read_text()) for name, path in MODULES.items()}
    owners = {}
    for name, path in MODULES.items():
        for exported in _exported_names(path):
            owners.setdefault(exported, set()).add(name)
    for name, tree in trees.items():
        for used in sorted(_loaded_names(tree) - _bound_names(tree) - BUILTINS):
            defined_in = sorted(owners.get(used, set()) - {name})
            if defined_in:
                offenders.append(f"{name}.py: uses {used} (defined in {defined_in}) without importing it")
    assert offenders == [], f"missing sibling imports: {offenders}"
