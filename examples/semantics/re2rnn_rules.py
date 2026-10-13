# RE2RNN's rule language and label aggregation, mirrored faithfully (Jiang et al., EMNLP 2020; github.com/jeffchy/RE2RNN):
#   - a config file lists "[label]" headers, each followed by one rule per line; "//" starts a comment;
#   - rules are word-level: "$" any token, "%" a number, "&" a punctuation token, words, "* + ? {m} {m,n}", "|", "( )";
#   - a sentence (with its BOS/EOS tokens) matches a rule only if the rule covers it entirely;
#   - a label scores the number of its rules that match, minus, for every "priority" (dominant, dominated) pair, the
#     number of matching rules of the dominant label; the predicted label is the arg-max, ties and no match going to
#     the label with the lowest index (as torch.argmax does in their code).
import re

# The punctuation set and number test of their fsa_to_tensor.py.
PUNCTUATIONS = {",", "，", ":", "：", "!", "！", "《", "》", "。", "；", ".", "(", ")", "（", "）", "|", "?", '"'}
CLASSES = {
    "<number>": lambda token: token.replace(".", "", 1).isdigit(),
    "<punct>": lambda token: token in PUNCTUATIONS,
}
# The priorities of their create_logic_mat_bias.py: (dominant label, dominated label).
PRIORITIES = {
    "ATIS": [("ground_fare", "ground_service"), ("flight", "city"), ("flight", "airport"), ("flight", "airline"), ("flight", "meal"),
             ("flight", "capacity"), ("flight", "cheapest"), ("flight", "distance"), ("airfare", "flight"), ("flight", "flight+airline"),
             ("abbreviation", "restriction"), ("ground_service", "airport")],
    "TREC": [("ENTITY", "DESCRIPTION")],
    "SMS": [("spam", "ham")],
}
# The rule file their loader uses for each dataset.
RULE_FILES = {"ATIS": "rules.v0.config", "TREC": "rules.old.config", "SMS": "rules.config"}

_TOKEN = re.compile(r"\{\d+(?:,\d+)?\}|[$%&*+?|()]|[A-Za-z'`]+")


def read_rules(text: str) -> list[tuple[str, str]]:
    # (label, rule) pairs in file order.
    rules, label = [], None
    for line in text.split("\n"):
        line = line.strip()
        header = re.fullmatch(r"\[([A-Za-z_+]+)\]", line)
        if header:
            label = header.group(1)
            continue
        rule = line.split("//")[0].strip()
        if rule and label is not None:
            rules.append((label, rule))
    return rules


def _expand_ranges(tokens: list[str]) -> list[str]:
    # X{m} → X repeated m times; X{m,n} → X m times followed by n-m nested optional copies: X{0,2} → ( X ( X ) ? ) ?
    out: list[str] = []
    for token in tokens:
        quantifier = re.fullmatch(r"\{(\d+)(?:,(\d+))?\}", token)
        if not quantifier:
            out.append(token)
            continue
        if out[-1] == ")":
            depth, begin = 0, len(out) - 1
            while True:
                depth += {")": 1, "(": -1}.get(out[begin], 0)
                if depth == 0:
                    break
                begin -= 1
        else:
            begin = len(out) - 1
        atom = out[begin:]
        del out[begin:]
        low = int(quantifier.group(1))
        high = int(quantifier.group(2)) if quantifier.group(2) else low
        out += atom * low
        optional: list[str] = []
        for _ in range(high - low):
            optional = ["(", *atom, *optional, ")", "?"]
        out += optional
    return out


def to_dre(rule: str) -> str:
    leftover = _TOKEN.sub("", rule).strip()
    if leftover:
        raise ValueError(f"Unsupported characters {leftover!r} in rule {rule!r}")
    tokens = _expand_ranges(_TOKEN.findall(rule))
    return " ".join({"$": ".", "%": "<number>", "&": "<punct>"}.get(token, token) for token in tokens)


def aggregate(rule_scores: list[list[float]], rule_labels: list[str], labels: list[str], dataset: str) -> list[int]:
    # rule_scores[s][r]: score of rule r on sentence s. Returns the predicted label index per sentence.
    index = {label: n for n, label in enumerate(labels)}
    predictions = []
    for scores in rule_scores:
        totals = [0.0] * len(labels)
        for score, label in zip(scores, rule_labels):
            totals[index[label]] += score
        adjusted = list(totals)
        for dominant, dominated in PRIORITIES[dataset]:
            adjusted[index[dominated]] -= totals[index[dominant]]
        predictions.append(max(range(len(labels)), key=lambda n: (adjusted[n], -n)))
    return predictions


def rules_by_label(rules: list[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for label, rule in rules:
        grouped.setdefault(label, []).append(rule)
    return grouped
