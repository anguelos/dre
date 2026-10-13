import pytest

from dre import tokenizing
from dre.optional import MissingDependency
from dre.tokenizing import naive_sentence_tokenize, naive_word_tokenize, words_of_sentences


def test_naive_sentence_tokenize():
    assert naive_sentence_tokenize("  Hello there. How are you?  Fine!") == ["Hello there.", "How are you?", "Fine!"]
    assert naive_sentence_tokenize("") == []


def test_naive_word_tokenize():
    assert naive_word_tokenize("Don't stop, well-known friend.") == ["Don't", "stop", ",", "well-known", "friend", "."]


def test_words_of_sentences_naive():
    assert list(words_of_sentences("Hi you. Bye!", tokenizer="naive")) == [["Hi", "you", "."], ["Bye", "!"]]


def test_words_of_sentences_auto_without_nltk(monkeypatch):
    monkeypatch.setattr(tokenizing, "nltk", MissingDependency("nltk"))
    with pytest.warns(UserWarning, match="pip install nltk"):
        assert list(words_of_sentences("Hi you. Bye!")) == [["Hi", "you", "."], ["Bye", "!"]]


def test_words_of_sentences_auto_nltk_without_data(monkeypatch):
    class NltkWithoutData:
        @staticmethod
        def word_tokenize(text):
            raise LookupError("punkt not found")

    monkeypatch.setattr(tokenizing, "nltk", NltkWithoutData)
    with pytest.warns(UserWarning, match="nltk.downloader"):
        assert list(words_of_sentences("Hi you. Bye!")) == [["Hi", "you", "."], ["Bye", "!"]]


def test_words_of_sentences_auto_with_nltk(monkeypatch):
    class FakeNltk:
        sent_tokenize = staticmethod(lambda text: [text])
        word_tokenize = staticmethod(lambda text: text.split())

    monkeypatch.setattr(tokenizing, "nltk", FakeNltk)
    assert list(words_of_sentences("Hi you. Bye!")) == [["Hi", "you.", "Bye!"]]


def test_words_of_sentences_nltk_missing(monkeypatch):
    monkeypatch.setattr(tokenizing, "nltk", MissingDependency("nltk"))
    with pytest.warns(UserWarning), pytest.raises(ImportError):
        list(words_of_sentences("Hi.", tokenizer="nltk"))


def test_words_of_sentences_unknown_tokenizer():
    with pytest.raises(ValueError):
        list(words_of_sentences("Hi.", tokenizer="spacy"))
