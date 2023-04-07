from typing import NamedTuple, Iterable

from core import NonTerminal, Rule

from rich import print as print_rich
from rich.pretty import pretty_repr
from rich.traceback import install

from parse_grammar import parse_grammar
from recognizers import Recognizer
from tokenizer import Tokenizer

install(show_locals=True)


class EarleyItem(NamedTuple):
    name: NonTerminal
    dot: int
    start: int
    rule: Rule

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.rule[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.rule[self.dot:])}     ({self.start})"
        )


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


class EarleyRecognizer(Recognizer):
    def recognizes(self, tokens: list[Tokenizer.Token]) -> bool:
        # initialize the recognizer; we have exactly one set for each token
        assert len(tokens) > 0, "Cannot recognize an empty string"
        assert tokens[-1].token_type == "eof", "Last token must be EOF"

        earley_sets = [EarleySet() for _ in range(len(tokens))]
        earley_sets[0].extend(
            EarleyItem(
                self.grammar.start_symbol,
                0,
                0,
                rule,
            )
            for rule in self.grammar[self.grammar.start_symbol]
        )
        for pos, (token, earley_set) in enumerate(zip(tokens, earley_sets)):
            index = 0
            while index < len(earley_set):
                print_rich(pretty_repr(earley_sets))
                name, dot, start, rule = earley_set[index]
                # We have not completed yet, so we can predict or scan
                if dot < len(rule):
                    right = rule[dot]
                    if isinstance(right, NonTerminal):
                        # We add the corresponding rules to the current state set.
                        earley_set.extend(
                            EarleyItem(right, 0, pos, rule)
                            for rule in self.grammar[right]
                        )
                    else:
                        if pos + 1 < len(earley_sets) and right.matches(token):
                            earley_sets[pos + 1].append(
                                EarleyItem(
                                    name,
                                    dot + 1,
                                    start,
                                    rule,
                                )
                            )
                else:
                    # completion
                    for item in earley_sets[start]:
                        if item.dot < len(item.rule):
                            if item.rule[item.dot] == name:
                                earley_set.append(
                                    EarleyItem(
                                        item.name,
                                        item.dot + 1,
                                        item.start,
                                        item.rule,
                                    )
                                )
                index += 1

        # are complete (the fat dot is at the end),
        # have started at the beginning (state set 0),
        # have the same name that has been chosen at the beginning ("Sum").
        return any(
            item.dot == len(item.rule)
            and item.start == 0
            and item.name == self.grammar.start_symbol
            for item in earley_sets[-1]
        )


if __name__ == "__main__":
    g = """
        <Program>
        <Program>       ::= <Sum>
        <Sum>           ::= <Sum> <PlusOrMinus> <Product> | <Product>
        <Product>       ::= <Product> <MulOrDiv> <Factor> | <Factor>
        <Factor>        ::= (<Sum>) | <Number>
        <Number>        ::= integer 
        <PlusOrMinus>   ::= + | -
        <MulOrDiv>      ::= * | /
    """

    table = {"(": "(", ")": ")", "+": "+", "-": "-", "*": "*", "/": "/"}

    g = parse_grammar(g, table)
    print_rich(pretty_repr(g))

    tks = Tokenizer("1 + (2*3 - 4)", table).get_tokens_no_whitespace()

    earley_recognizer = EarleyRecognizer(g)
    print_rich(pretty_repr(earley_recognizer.recognizes(tks)))
