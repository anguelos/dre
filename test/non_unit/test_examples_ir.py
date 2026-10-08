import subprocess
import sys
from pathlib import Path

import pytest

IR_DIR = Path(__file__).resolve().parents[2] / "examples" / "ir"
sys.path.insert(0, str(IR_DIR))

import trec_io  # noqa: E402
from word_regex import compile_word_regex, matches_empty, may_match, query_tokens, required_words, token_string  # noqa: E402


@pytest.mark.parametrize(("query", "text", "expected"), [
    ("federal register", "The Federal Register, vol. 59", True),
    ("federal register", "federal  income register", False),
    ("federal . register", "federal income register", True),
    ("federal . register", "federal register", False),
    ("federal income ? register", "federal register", True),
    ("federal income ? register", "federal income register", True),
    ("federal ( income | tax ) register", "federal tax register", True),
    ("federal ( income | tax ) register", "federal fee register", False),
    ("a b +", "a b b b", True),
    ("a x * b", "a b", True),
    ("^ the", "the end", True),
    ("^ end", "the end", False),
    ("end $", "the end", True),
    ("reg", "register", False),
    ("register", "preregister", False),
])
def test_word_regex(query, text, expected):
    assert (compile_word_regex(query).search(token_string(text)) is not None) == expected


def test_word_regex_errors_and_empty():
    with pytest.raises(ValueError):
        query_tokens("cfr 40")
    assert query_tokens("a(b|c)?") == ["a", "(", "b", "|", "c", ")", "?"]
    assert matches_empty("a *")
    assert not matches_empty("a b")


@pytest.mark.parametrize(("query", "required"), [
    ("federal register", ["federal", "register"]),
    ("federal . register", ["federal", "register"]),
    ("federal income ? register", ["federal", "register"]),
    ("a x * b +", ["a", "b"]),
    ("federal ( income | tax ) register", ["federal", "register"]),
    ("federal | register", []),
    ("^ The end $", ["the", "end"]),
])
def test_required_words(query, required):
    assert required_words(query) == required


@pytest.mark.parametrize("query", ["federal register", "federal ( income | tax ) register", "a x * b +", "in ( the | afrs ) federal", "to purchase ? the"])
@pytest.mark.parametrize("text", ["the federal tax register", "federal register", "a b b", "in the federal", "to the", "nothing here"])
def test_prefilter_never_drops_a_match(query, text):
    if compile_word_regex(query).search(token_string(text)):
        assert may_match(required_words(query), " " + token_string(text))


def test_trec_io_roundtrip(tmp_path):
    queries = [trec_io.Query("q1", "word", "alpha"), trec_io.Query("q2", "pattern", "a ( b | c ) d")]
    trec_io.write_queries(tmp_path / "q.tsv", queries)
    assert trec_io.read_queries(tmp_path / "q.tsv") == queries
    qrels = {"q1": {"D2": 1, "D1": 2}}
    trec_io.write_qrels(tmp_path / "qrels", qrels)
    assert trec_io.read_qrels(tmp_path / "qrels") == qrels
    trec_io.write_docs(tmp_path / "d.jsonl", [("D1", "text one"), ("D2", "ünïcode")])
    assert list(trec_io.read_docs(tmp_path / "d.jsonl")) == [("D1", "text one"), ("D2", "ünïcode")]
    trec_io.write_token_strings(tmp_path / "t.tsv", [("D1", "a b ")])
    assert list(trec_io.read_token_strings(tmp_path / "t.tsv")) == [("D1", "a b ")]
    assert not (tmp_path / "t.tsv.part").exists()


def test_write_run_orders_by_score_then_docno(tmp_path):
    trec_io.write_run(tmp_path / "r.run", {"q1": [("B", 1.0), ("C", 2.0), ("A", 1.0)]}, tag="t", depth=2)
    lines = (tmp_path / "r.run").read_text().splitlines()
    assert lines == ["q1 Q0 C 1 2 t", "q1 Q0 A 2 1 t"]
    assert trec_io.read_run(tmp_path / "r.run") == {"q1": [("C", 2.0), ("A", 1.0)]}


def test_iter_trec_sgml():
    text = "<DOC>\n<DOCNO> FR1 </DOCNO>\n<TEXT><!-- PJG x -->Hello <B>world</B></TEXT>\n</DOC>\n<DOC><DOCNO>FR2</DOCNO>Bye</DOC>"
    docs = list(trec_io.iter_trec_sgml(text))
    assert [d for d, _ in docs] == ["FR1", "FR2"]
    assert docs[0][1].split() == ["Hello", "world"]
    with pytest.raises(ValueError):
        list(trec_io.iter_trec_sgml("<DOC>no docno</DOC>"))


def test_synthetic_pipeline(tmp_path):
    def make(*targets):
        result = subprocess.run(["make", f"DATA={tmp_path}", "COLLECTION=synthetic", *targets], cwd=IR_DIR, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    missing = subprocess.run(["make", f"DATA={tmp_path}", "COLLECTION=synthetic", "compile"], cwd=IR_DIR, capture_output=True, text=True)
    assert missing.returncode != 0 and "make download" in missing.stderr
    make("download")
    make("compile")
    make("run_re", "run_fuzzy")
    assert "Nothing to be done" in make("compile", "run_re").stdout
    root = tmp_path / "synthetic"
    qrels = trec_io.read_qrels(root / "qrels.txt")
    queries = trec_io.read_queries(root / "queries.tsv")
    assert {q.family for q in queries} == {"word", "phrase", "pattern"}
    assert set(qrels) == {q.qid for q in queries}
    assert (root / "qrels.dropped.tsv").read_text() == ""
    original = trec_io.read_run(root / "runs" / "re" / "original.run")
    assert {qid: {d for d, _ in docs} for qid, docs in original.items() if docs} == {qid: set(docs) for qid, docs in qrels.items()}
    assert (root / "runs" / "re" / "degrade20.run").exists()
    make("run_dre_exact", "run_phoc_soft")
    for version in ["original", "degrade5", "degrade20"]:
        exact = trec_io.read_run(root / "runs" / "re" / f"{version}.run")
        dre = trec_io.read_run(root / "runs" / "dre_exact" / f"{version}.run")
        assert {q: {d for d, _ in docs} for q, docs in dre.items()} == {q: {d for d, _ in docs} for q, docs in exact.items() if docs}
    phoc_original = trec_io.read_run(root / "runs" / "phoc_soft" / "original.run")
    for qid, relevant in qrels.items():
        assert set(relevant) <= {docno for docno, _ in phoc_original[qid]}
    fuzzy = trec_io.read_run(root / "runs" / "fuzzy" / "original.run")
    for qid, relevant in qrels.items():
        exact = {docno for docno, score in fuzzy[qid] if score == 0}
        assert set(relevant) <= exact
    try:
        import ir_measures  # noqa: F401
    except ImportError:
        pass
    else:
        make("eval_re")
        rows = (root / "eval" / "re" / "original.tsv").read_text().splitlines()
        assert rows[-1].startswith("mean\tall\t1.0000")
    make("clean_re")
    assert not (root / "runs" / "re").exists()
    make("run_re", "VERSIONS=degrade5")
    make("clean")
    assert not (root / "runs").exists() and (root / "raw" / "SHA256SUMS").exists() and (root / "qrels.txt").exists()
    make("mrproper")
    assert not tmp_path.exists()
