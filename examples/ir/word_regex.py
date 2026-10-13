# Exact matching of dre word-level regexes with Python `re`: the oracle for qrels and the exact-match baseline.
# A document becomes a token string where every token is followed by one space, and every query word becomes
# `word ` — so concatenation, alternation and quantifiers keep their meaning at token granularity.
import re

from dre.tokenizing import naive_word_tokenize

_QUERY_TOKEN = re.compile(r"[A-Za-z]+|[.^$*?+|()]")
_VALID_QUERY = re.compile(r"(?:\s*(?:[A-Za-z]+|[.^$*?+|()]))*\s*")


def query_tokens(query: str) -> list[str]:
    if _VALID_QUERY.fullmatch(query) is None:
        raise ValueError(f"Not a word regex: {query!r}")
    return _QUERY_TOKEN.findall(query)


def compile_word_regex(query: str) -> re.Pattern:
    return re.compile(r"(?<![^ ])(?:" + word_regex_body(query) + ")")


def word_regex_body(query: str) -> str:
    parts = []
    for token in query_tokens(query):
        # Each token unit is grouped so that a following quantifier applies to the whole token, trailing space included.
        if token == ".":
            parts.append(r"(?:[^ ]+ )")
        elif token == "(":
            parts.append("(?:")
        elif token in ")*?+|^$":
            parts.append(token)
        else:
            parts.append("(?:" + re.escape(token.lower()) + " )")
    return "".join(parts)


def required_words(query: str) -> list[str]:
    # Words every match must contain: top-level words not made optional by a following ? or *. A top-level |
    # makes nothing required; words inside groups are conservatively ignored.
    tokens = query_tokens(query)
    depth, required = 0, []
    for n, token in enumerate(tokens):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif token == "|" and depth == 0:
            return []
        elif depth == 0 and token.isalpha() and (n + 1 == len(tokens) or tokens[n + 1] not in "?*"):
            required.append(token.lower())
    return required


def may_match(required: list[str], padded_token_string: str) -> bool:
    # padded_token_string is " " + token_string(...): every token is then delimited by spaces on both sides.
    return all(f" {word} " in padded_token_string for word in required)


def token_string(text: str) -> str:
    return "".join(token.lower() + " " for token in naive_word_tokenize(text))


def matches_empty(query: str) -> bool:
    return compile_word_regex(query).search("") is not None
