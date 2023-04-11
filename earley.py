from typing import Iterable, NamedTuple

from rich.traceback import install

from core import NonTerminal, Rule
from tokenizer import Token

install(show_locals=True)


class EarleyItem(NamedTuple):
    name: NonTerminal
    dot: int
    explicit_index: int
    rule: Rule

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.rule[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.rule[self.dot:])}     ({self.explicit_index})"
        )

    def advance(self):
        return EarleyItem(self.name, self.dot + 1, self.explicit_index, self.rule)

    def is_completed(self):
        return self.dot >= len(self.rule)


class EarleySet(list[EarleyItem]):
    def __init__(self, *items):
        assert all(
            isinstance(item, EarleyItem) for item in items
        ), "All items must be EarleyItem"
        super().__init__(set(items))

    def append(self, earley_item: EarleyItem) -> None:
        if not isinstance(earley_item, EarleyItem):
            raise TypeError(f"Expected EarleyItem, got {type(earley_item)}")
        if earley_item not in self:
            super().append(earley_item)

    def extend(self, earley_items: Iterable[EarleyItem]) -> None:
        for earley_item in earley_items:
            self.append(earley_item)

    def remove_unfinished(self):
        return EarleySet(*(item for item in self if item.is_completed()))


def gen_early_sets(grammar, tokens: list[Token]) -> list[EarleySet]:
    # initialize the recognizer; we have exactly one set for each token
    assert len(tokens) > 0, "Cannot recognize an empty string"
    assert tokens[-1].token_type == "eof", "Last token must be EOF"

    nullable_set = grammar.nullable()

    earley_sets = [EarleySet() for _ in range(len(tokens))]
    earley_sets[0].extend(
        EarleyItem(
            grammar.start_symbol,
            0,
            0,
            rule,
        )
        for rule in grammar[grammar.start_symbol]
    )
    for pos, (token, earley_set) in enumerate(zip(tokens, earley_sets)):
        current_pos = 0
        while current_pos < len(earley_set):
            name, dot, start, rule = earley_set[current_pos]

            if dot < len(rule):
                right = rule[dot]
                if isinstance(right, NonTerminal):
                    if right in nullable_set:
                        earley_set.append(earley_set[current_pos].advance())

                    earley_set.extend(
                        EarleyItem(right, 0, pos, rule) for rule in grammar[right]
                    )
                elif pos + 1 < len(earley_sets) and right.matches(token):
                    earley_sets[pos + 1].append(earley_set[current_pos].advance())
            else:
                for item in earley_sets[start]:
                    if not item.is_completed() and item.rule[item.dot] == name:
                        earley_set.append(item.advance())
            current_pos += 1

    return earley_sets
