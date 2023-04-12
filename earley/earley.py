from typing import NamedTuple

from rich.traceback import install

from grammar import Expansion, NonTerminal, Terminal
from lr import LRState
from utils import Token

install(show_locals=True)


class EarleyItem(NamedTuple):
    name: NonTerminal
    dot: int
    explicit_index: int
    rule: Expansion

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.rule[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.rule[self.dot:])}     ({self.explicit_index})"
        )

    def advance(self):
        return EarleyItem(self.name, self.dot + 1, self.explicit_index, self.rule)

    def completed(self):
        return self.dot >= len(self.rule)


def gen_early_sets(grammar, tokens: list[Token]) -> list[LRState[EarleyItem]]:
    # initialize the recognizer; we have exactly one set for each token
    assert len(tokens) > 0, "Cannot recognize an empty string"
    assert tokens[-1].token_type == "eof", "Last token must be EOF"

    nullable_set = grammar.gen_nullable()

    earley_sets = [LRState[EarleyItem](cls=EarleyItem) for _ in range(len(tokens))]
    earley_sets[0].extend(
        EarleyItem(
            grammar.start,
            0,
            0,
            rule,
        )
        for rule in grammar[grammar.start]
    )
    for pos, (token, earley_set) in enumerate(zip(tokens, earley_sets)):
        current_pos = 0
        while current_pos < len(earley_set):
            name, dot, start, rule = earley_set[current_pos]

            if dot < len(rule):
                right = rule[dot]
                if isinstance(right, Terminal):
                    if pos + 1 < len(earley_sets) and right.matches(token):
                        earley_sets[pos + 1].append(earley_set[current_pos].advance())
                elif isinstance(right, NonTerminal):
                    if right in nullable_set:
                        earley_set.append(earley_set[current_pos].advance())

                    earley_set.extend(
                        EarleyItem(right, 0, pos, rule) for rule in grammar[right]
                    )
            else:
                for item in earley_sets[start]:
                    if not item.completed() and item.rule[item.dot] == name:
                        earley_set.append(item.advance())
            current_pos += 1

    return earley_sets
