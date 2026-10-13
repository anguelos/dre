# Optional dependencies: heavy packages that only some functions, scripts or examples need. Importing them never fails:
# a missing one is replaced by a placeholder, and only code that actually uses it prints how to install it (pip, and
# apt for system packages) and raises ImportError, or warns and falls back where a fallback exists.
# Every optional dependency is listed here and in setup.py's "recommended" extra: pip install dre[recommended].
import importlib
import warnings
from types import ModuleType

# module name -> (pip distribution, apt packages it needs at run time, what needs it)
OPTIONAL = {
    "nltk": ("nltk", (), "NLTK sentence and word tokenization (dre.words_of_sentences)"),
    "graphviz": ("graphviz", ("graphviz",), "drawing automata (FAGraph.plot, bin/word_regex_viewer)"),
    "ir_measures": ("ir_measures", (), "IR evaluation (examples/ir)"),
    "regex": ("regex", (), "the fuzzy regex baseline (examples/ir)"),
    "pyterrier": ("python-terrier", ("default-jre-headless",), "the BM25 baselines (examples/ir)"),
    "pandas": ("pandas", (), "the BM25 baselines (examples/ir)"),
    "rapidfuzz": ("rapidfuzz", (), "Levenshtein lookup baselines (examples/phoc_semantic)"),
    "fasttext": ("fasttext", ("build-essential", "python3-dev"), "the fastText baseline (examples/phoc_semantic)"),
    "tensorboard": ("tensorboard", (), "TensorBoard logging (torch.utils.tensorboard; examples/phoc)"),
}


def install_instructions(name: str) -> str:
    # name: a declared optional package (or one of its modules, "rapidfuzz.distance")
    pip_name, apt_packages, needed_by = OPTIONAL[name.split(".")[0]]
    message = f"{needed_by} needs the optional package {pip_name!r}: pip install {pip_name}  (or all of them: pip install dre[recommended])"
    if apt_packages:
        message += f"; it also needs: sudo apt install {' '.join(apt_packages)}"
    return message


def warn_missing(name: str, fallback: str) -> None:
    warnings.warn(f"{install_instructions(name)}. Falling back to {fallback}.", stacklevel=3)


class MissingDependency(ModuleType):
    # Stands in for a missing optional module: false in a boolean test, and any use warns with the install
    # instructions and raises ImportError.
    def __init__(self, name: str, package: str | None = None) -> None:
        super().__init__(name)
        self._dre_missing = name if package is None else package

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, attribute: str):
        if attribute.startswith("__"):  # introspection (copy, pickle, doctest, ...) must not trigger the warning
            raise AttributeError(attribute)
        message = install_instructions(self._dre_missing)
        warnings.warn(message, stacklevel=2)
        raise ImportError(message)


def optional_import(name: str, package: str | None = None) -> ModuleType:
    # The module (or a submodule, e.g. "rapidfuzz.distance") if installed, otherwise a MissingDependency. package: the
    # declared optional dependency that provides the module when it is not the module's own top-level package (e.g.
    # "torch.utils.tensorboard" needs "tensorboard"); it must be listed in OPTIONAL.
    package = name.split(".")[0] if package is None else package
    if package not in OPTIONAL:
        raise KeyError(f"{package!r} is not a declared optional dependency; add it to dre.optional.OPTIONAL and setup.py")
    try:
        return importlib.import_module(name)
    except ImportError:
        return MissingDependency(name, package)
