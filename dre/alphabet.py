import torch
from typing import List, Union, Dict


class Alphabet:
    @staticmethod
    def load_alphabet_from_string(s: str):
        """Instatiate an alphabet from a string

        Args:
            s (str): either a TSV representation of the alphabet or a string
            where each character is a symbol, their order dicates their code
            and code 0 maps to the empty string"

        Returns:
            Alphabet: the alphabet object
        """
        if "\t" in s:
            return Alphabet(tsv_repr=s)
        else:
            return Alphabet(symbol_sequence=s)

    def __init__(self, symbol_sequence: str = "",
                 code2symbol: Union[Dict[int, str], None] = None,
                 symbol2code: Union[Dict[str, int], None] = None,
                 tsv_repr: str = "", unknown_symbol: str = ""):
        self.unknown_symbol = unknown_symbol
        if symbol_sequence != "":
            assert code2symbol is None and symbol2code is None and tsv_repr == ""
            self.code2symbol = {0: self.unknown_symbol}
            self.symbol2code = {self.unknown_symbol: 0}  # 0 is reserved for unknown symbols
            for n, s in enumerate(symbol_sequence):
                n = n + 1
                self.code2symbol[n] = s
                self.symbol2code[s] = n
        elif tsv_repr != "":
            assert len(code2symbol) == 0 and len(symbol2code) == 0 and symbol_sequence == ""
            code_values = [l.split("\t") for l in tsv_repr.strip().split("\n")]
            self.code2symbol = {int(c.strip()): s.strip() for c, s in code_values}
            self.symbol2code = {s: int(c) for c, s in self.code2symbol.items()}
            #  Sanity check, no duplicate symbols or codes
            assert len(self.code2symbol) == len(self.symbol2code) == len(code_values)
        elif code2symbol is not None and symbol2code is None:
            self.code2symbol = {k: v for k, v in code2symbol.items()}
            self.symbol2code = {v: k for k, v in code2symbol.items()}
        elif code2symbol is None and symbol2code is not None:
            self.symbol2code = {k: v for k, v in symbol2code.items()}
            self.code2symbol = {v: k for k, v in symbol2code.items()}
        elif code2symbol is not None and symbol2code is not None:
            self.code2symbol = {k: v for k, v in code2symbol.items()}
            self.symbol2code = {k: v for k, v in symbol2code.items()}
        else:
            raise ValueError("Either symbol_sequence or tsv_repr or code2symbol or symbol2code must be provided")
        assert set(self.code2symbol.values()) == set(self.symbol2code.keys()) and set(self.code2symbol.keys()) == set(self.symbol2code.values())

    def encode_symbol(self, s: str) -> int:
        return self.symbol2code.get(s, 0)

    def decode_symbol(self, s: int) -> str:
        return self.code2symbol.get(s, self.unknown_symbol)

    def encode_sequence(self, s: str) -> torch.Tensor:
        return torch.tensor([self.encode_symbol(c) for c in s], dtype=torch.long)

    def decode_sequence(self, s: torch.Tensor) -> str:
        return "".join([self.decode_symbol(c) for c in s.cpu().tolist()])

    def encode_sequences(self, s: List[str]) -> torch.Tensor:
        sequences = [self.encode_sequence(seq) for seq in s]
        lengths = [s.size(0) for s in sequences]
        max_len = max(lengths)
        res = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for n, seq in enumerate(sequences):
            res[n, :seq.size(0)] = seq
        return res

    def decode_sequences(self, s: torch.Tensor) -> List[str]:
        res = []
        for n in range(s.size(0)):
            res.append(self.decode_sequence(s[n, :]))
        return res

    def __str__(self):
        """Generates a TSV version of the alphabet

        Returns:
            str: a TSV representation of the alphabet the first column is the code and the second column is the symbol
        """
        return "\n".join([f"{k}\t{v}" for k, v in sorted(self.code2symbol.items())])

    def __repr__(self):
        if list(range(len(self))) == list(sorted(self.code2symbol.keys())) and all([len(v) <= 1 for v in self.code2symbol.values()]):
            res = sorted([(k, v) for k, v in self.code2symbol.items()])
            res = "".join([v for _, v in res])
            return f"Alphabet(symbol_sequence={repr(res)}, unknown_symbol={repr(self.unknown_symbol)})"
        else:  # The representation is not a simple sequence of characters
            return f"Alphabet(tsv_repr={repr(str(self))}, unknown_symbol={repr(self.unknown_symbol)})"

    def __get_item__(self, v):
        if isinstance(v, int):
            return self.decode_symbol(v)
        elif isinstance(v, str):
            return self.encode_symbol(v)
        else:
            raise ValueError("v must be either a string or an integer")

    def __len__(self):
        return len(self.code2symbol)
