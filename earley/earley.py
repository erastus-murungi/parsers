from typing import NamedTuple, Optional, cast

from grammar import Expansion, Grammar, NonTerminal, Terminal
from lr import LRState
from utils import Token


class EarleyItem(NamedTuple):
    name: NonTerminal
    dot: int
    explicit_index: int
    rule: Expansion

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.rule[:self.dot])}"
            f" ● "
            f"{' '.join(repr(sym) for sym in self.rule[self.dot:])}     ({self.explicit_index})"
        )

    def advance(self):
        return EarleyItem(self.name, self.dot + 1, self.explicit_index, self.rule)

    def completed(self):
        return self.dot >= len(self.rule)


class EarleyError(SyntaxError):
    def __init__(
        self, expected_terminals: list[Terminal], failure_token: Token, source: str
    ):
        self.expected_terminals: list[Terminal] = expected_terminals
        ident = "\t\t"
        super().__init__(
            f"@ {failure_token.loc}:\n"
            f" > {source}\n"
            f" > {'-'*failure_token.loc.offset}\n"
            + "Expected one of:\n"
            + "\n".join(f"{ident} {tok!s}" for tok in expected_terminals)
            + f"\nbut encountered: {failure_token}"
        )


def gen_earley_sets(
    grammar: Grammar, tokens: list[Token], source: str
) -> list[LRState[EarleyItem]]:
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
        if not earley_set:
            raise EarleyError(
                [
                    cast(Terminal, rule[dot])
                    for _, dot, _, rule in earley_sets[pos - 1]
                    if dot < len(rule) and isinstance(rule[dot], Terminal)
                ],
                tokens[pos - 1],
                source,
            )
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

    # we look for an items that:
    #   1) are complete (the fat dot is at the end),
    #   2) have started at the beginning (state set 0),
    #   3) have the same name that has been chosen at the beginning ("Sum").
    #   4) are in the last set (the last token has been consumed)
    if not any(
        item.dot == len(item.rule)
        and item.explicit_index == 0
        and item.name == grammar.start
        for item in earley_sets[-1]
    ):
        raise EarleyError(
            [
                cast(Terminal, rule[dot])
                for _, dot, _, rule in earley_sets[-1]
                if dot < len(rule) and isinstance(rule[dot], Terminal)
            ],
            tokens[-1],
            source,
        )
    return earley_sets
