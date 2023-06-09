from dataclasses import dataclass
from functools import cache

from more_itertools import one

from grammar import EOF, Grammar, NonTerminal, Terminal
from lr.core import LRState, Reduce
from lr.lr0 import LR0Item, LR0ParsingTable


@dataclass(frozen=True, slots=True, eq=True)
class LR1Item(LR0Item):
    lookahead: Terminal

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.expansion[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.expansion[self.dot:])}         ({self.lookahead!r})"
        )

    def __str__(self):
        return (
            f"{self.name!s} -> {' '.join(str(sym) for sym in self.expansion[:self.dot])}"
            f" . "
            f"{' '.join(str(sym) for sym in self.expansion[self.dot:])}          ({self.lookahead!s})"
        )

    def __iter__(self):
        yield from [self.name, self.dot, self.expansion, self.lookahead]

    def advance(self):
        return LR1Item(self.name, self.dot + 1, self.expansion, self.lookahead)


class LR1ParsingTable(LR0ParsingTable):
    def init_kernel(self):
        # append EOF to the start symbol
        return LRState[LR1Item](
            LR1Item(
                self.grammar.start,
                0,
                one(self.grammar[self.grammar.start]).append(EOF),
                EOF,  # could be anything
            ),
            cls=LR1Item,
        )

    @cache
    def closure(self, configuration_set: LRState[LR1Item]) -> LRState[LR1Item]:
        """
        Compute the closure of LR(1) item set
        :param configuration_set: a set of LR(1) items
        :return: closure of the set
        """

        items = configuration_set.copy()
        while True:
            initial_closure_size = len(items)
            for _, dot, rule, lookahead in items.yield_unfinished():
                b, beta = rule[dot], rule[dot + 1 :]
                if isinstance(b, NonTerminal):
                    for gamma in self.grammar[b]:
                        for w in self.grammar.first(beta + (lookahead,)):
                            items.append(LR1Item(b, 0, gamma, w))
            if len(items) == initial_closure_size:
                break
        return items

    def compute_reduce_actions(self):
        for state in self.states:
            for name, _, rule, lookahead in state.yield_finished():
                if (state, lookahead.name) not in self:
                    self[(state, lookahead.name)] = Reduce(name, len(rule))
                else:
                    raise ValueError(
                        f"Encountered shift/reduce conflict on \n"
                        f" state: {str(state)}\n and symbol: {lookahead.name}\n"
                        f"  {self[(state, lookahead.name)]} and \n"
                        f"  Reduce({name!s} -> {rule!s})"
                    )


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR1

    # table = {
    #     "x": "x",
    #     "(": "(",
    #     ")": ")",
    #     ",": ",",
    # }
    # g = """
    #         <S'>
    #         <S'> -> <S>
    #         <S> -> (<L>)
    #         <L> -> <S>
    #         <S> -> x
    #         <L> -> <L>,<S>
    # """
    # table = {"+": "+", "(": "(", ")": ")"}
    #
    # g = """
    #     <S>
    #     <S> -> <E>
    #     <E> -> <E> + <T>
    #     <E> -> <T>
    #     <T> -> ( <E> )
    #     <T> -> integer
    # """

    cfg = Grammar.from_str(*GRAMMAR1)
    print_rich(pretty_repr(cfg))

    p = LR1ParsingTable(cfg)
    print_rich(p.to_pretty_table())

    print_rich(pretty_repr(LR1ParsingTable(cfg)))
