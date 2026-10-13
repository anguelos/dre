import ast
import importlib
from pathlib import Path

import pytest

import dre.tokenizing
from dre.optional import OPTIONAL, MissingDependency, install_instructions, optional_import

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def nothing_installed(monkeypatch):
    def fail(name, *args, **kwargs):
        raise ImportError(name)
    monkeypatch.setattr(importlib, "import_module", fail)


def test_missing_dependency_warns_with_install_commands_when_used(nothing_installed):
    graphviz = optional_import("graphviz")
    assert isinstance(graphviz, MissingDependency) and not graphviz
    with pytest.warns(UserWarning, match="pip install graphviz") as warned, pytest.raises(ImportError, match="sudo apt install graphviz"):
        graphviz.Digraph("x")
    assert "dre[recommended]" in str(warned[0].message)


def test_submodules_and_undeclared_packages(nothing_installed):
    assert "pip install rapidfuzz" in install_instructions("rapidfuzz.distance")
    assert not optional_import("rapidfuzz.distance")
    with pytest.raises(KeyError):
        optional_import("not_a_declared_dependency")


def test_tokenizer_falls_back_with_a_warning(monkeypatch):
    monkeypatch.setattr(dre.tokenizing, "nltk", MissingDependency("nltk"))
    with pytest.warns(UserWarning, match="pip install nltk"):
        assert list(dre.tokenizing.words_of_sentences("Hello world. Bye.")) == [["Hello", "world", "."], ["Bye", "."]]
    with pytest.warns(UserWarning), pytest.raises(ImportError):
        list(dre.tokenizing.words_of_sentences("Hello.", tokenizer="nltk"))


def test_installed_packages_are_returned_as_they_are():
    regex = pytest.importorskip("regex")
    assert optional_import("regex") is regex


def test_optional_packages_are_only_imported_through_optional_import():
    sources = [p for directory in ["src", "examples", "bin"] for p in (ROOT / directory).rglob("*")
               if p.is_file() and (p.suffix == ".py" or p.parent.name == "bin") and "__pycache__" not in p.parts and "data" not in p.parts]
    direct = []
    for path in sources:
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            direct += [f"{path.relative_to(ROOT)}: {name}" for name in names if name.split(".")[0] in OPTIONAL]
    assert direct == []
