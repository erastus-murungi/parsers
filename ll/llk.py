from typing import Optional

from prettytable import PrettyTable

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalSequence
from ll.decidability import compute_k
from ll.first_k import first_k
from ll.follow_k import follow_k

LLKKey = tuple[NonTerminal, TerminalSequence]


class LLKParsingTable(dict[LLKKey, Expansion]):
    def __init__(self, grammar: Grammar, max_k: int = 10):
        super().__init__()
        k = compute_k(grammar, max_k)
        if k is None:
            raise ValueError(f"grammar is not LL(k) for k <= {max_k}")
        self.k = k
        self.grammar: Grammar = grammar
        self.construct()

    def choose_rule(
        self, origin: NonTerminal, token_index: int, tokens: list[Terminal]
    ) -> Optional[Expansion]:
        max_lookahead = min(self.k, len(tokens) - token_index)
        for lookahead in range(max_lookahead, 0, -1):
            ts = TerminalSequence(
                tokens[token_index : token_index + lookahead], lookahead
            )
            if (origin, ts) in self:
                return self[(origin, ts)]
        return None

    def get_expected(self, origin: NonTerminal) -> list[str]:
        expected: set[TerminalSequence] = set()
        for (origin_, terminal_sequence), rule in self.items():
            if origin == origin_:
                expected.add(terminal_sequence)
        return [str(ts) for ts in expected]

    def construct(self):
        FOLLOW = follow_k(self.grammar, self.k)[1]
        FIRST = first_k(self.grammar, self.k)
        for origin, expansion in self.grammar.iter_productions():
            first = FIRST[expansion]
            for ts in first:
                if not ts.is_eps():
                    self[(origin, ts)] = expansion
            if any(ts.is_eps() for ts in first):
                for ts in FOLLOW[origin]:
                    self[(origin, ts)] = expansion

    def __getitem__(self, key: LLKKey) -> Expansion:
        return super().__getitem__(key)

    def __setitem__(self, key: LLKKey, rule: Expansion):
        origin, terminal_id = key
        if (origin, terminal_id) in self:
            raise ValueError(
                f"grammar not LL({self.k}); "
                f"we have <{origin}, {terminal_id}> "
                f"mapping to {super().__getitem__((origin, terminal_id))!s}, {rule!s}"
            )
        super().__setitem__((origin, terminal_id), rule)

    def to_pretty_table(self) -> PrettyTable:
        table = PrettyTable()

        lookaheads = set(terminal_sequence for _, terminal_sequence in self.keys())
        table.field_names = [f" k={self.k} lookaheads"] + list(map(str, lookaheads))

        for origin in self.grammar.non_terminals:
            row: list[str] = [str(origin)]
            for terminal_sequence in lookaheads:
                if (origin, terminal_sequence) in self:
                    row.append(str(self[(origin, terminal_sequence)]))
                else:
                    row.append("error")
            table.add_row(row)

        return table

    def __str__(self):
        return self.to_pretty_table()


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR_JSON

    g = Grammar.from_str(*GRAMMAR_JSON)
    rich_print(pretty_repr(g))
