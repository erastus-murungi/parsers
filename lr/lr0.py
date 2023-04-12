from functools import cache
from typing import NamedTuple

from grammar.core import EMPTY, EOF, NonTerminal, Rule, Symbol, Terminal
from lr.core import Accept, Goto, LRTable, Reduce, Shift, State


class LR0Item(NamedTuple):
    name: NonTerminal
    dot: int
    rule: Rule

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.rule[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.rule[self.dot:])}"
        )

    def __str__(self):
        return (
            f"{self.name!s} -> {' '.join(str(sym) for sym in self.rule[:self.dot])}"
            f" . "
            f"{' '.join(str(sym) for sym in self.rule[self.dot:])}"
        )

    def advance(self):
        return LR0Item(self.name, self.dot + 1, self.rule)

    def completed(self):
        return self.dot >= len(self.rule)

    @property
    def at_start(self):
        return self.dot == 0


class LR0ParsingTable(LRTable[LR0Item]):
    def __init__(
        self,
        grammar,
        *,
        reduce: bool = False,
    ):
        super().__init__(grammar, reduce=reduce)

    @cache
    def closure(self, state: State[LR0Item]):
        # Closure adds more items to a set of items when
        # there is a dot to the left of a non-terminal;
        # The LR(0) closure step adds all items of the form B → •γ to a state
        # whenever the state contains an item A → α • Bβ.

        new_items: State[LR0Item] = state.copy()
        # for all items of the form A → α • Bβ
        changing = True
        while changing:
            changing = False
            for _, dot, rule in new_items.yield_unfinished():
                b = rule[dot]  # α • Bβ
                if isinstance(b, NonTerminal):
                    # add all items of the form B → •γ
                    initial_size = len(new_items)
                    new_items.extend(LR0Item(b, 0, gamma) for gamma in self.grammar[b])
                    changing = len(new_items) != initial_size
        return new_items

    @cache
    def goto(self, state: State[LR0Item], sym: Symbol) -> State[LR0Item]:
        # Goto is the transition function of the LR(0) automaton.
        # It takes a state and a symbol and returns the state that
        # results from shifting the dot over the symbol.
        # If the symbol is a terminal, then the state is unchanged.
        # If the symbol is a non-terminal, then the state is the closure
        # of the state resulting from shifting the dot over the symbol.
        assert sym is not EMPTY and sym is not EOF
        kernel = State[LR0Item](cls=LR0Item)
        for item in state.yield_unfinished():
            if item.rule[item.dot] == sym:
                kernel.append(item.advance())
        return self.closure(kernel)

    @cache
    def init_kernel(self):
        return State[LR0Item](
            LR0Item(
                self.grammar.start_symbol,
                0,
                self.grammar[self.grammar.start_symbol][0].append_marker(EOF),
            ),
            cls=LR0Item,
        )

    def compute_reduce_actions(self):
        for state in self.states:
            for item in state.yield_finished():
                for symbol in self.grammar.terminals:
                    if (state, symbol.id) not in self:
                        self[(state, symbol.id)] = Reduce(item.name, len(item.rule))
                    else:
                        raise ValueError(
                            f"Encountered shift/reduce conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.id}\n"
                            f"  {self[(state, symbol.id)]} and \n"
                            f"  Reduce({item.name!s} -> {item.rule!s})"
                        )

    def construct(self):
        states = {self.closure(self.init_kernel()): None}
        changing = True

        while changing:
            changing = False
            for state in states.copy():
                for _, dot, rule in state.yield_unfinished():
                    symbol = rule[dot]
                    if symbol is EOF:
                        # accept action
                        self[(state, symbol.id)] = Accept()
                        self.accept = (state, symbol.id)
                    else:
                        # shift action
                        target = self.goto(state, symbol)
                        assert target
                        if target not in states:
                            changing = True
                            states[target] = None
                        action = (
                            Shift(target)
                            if isinstance(symbol, Terminal)
                            else Goto(target)
                        )
                        if action != self.get((state, symbol.id), None):
                            self[(state, symbol.id)] = action
                            changing = True

        self.states = list(states)
        if self.reduce:
            self.compute_reduce_actions()


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr

    from utils.parse_grammar import parse_grammar

    table = {
        "x": "x",
        "(": "(",
        ")": ")",
        ",": ",",
    }
    g = """
            <S'>
            <S'> -> <S>
            <S> -> (<L>)
            <L> -> <S>
            <S> -> x
            <L> -> <L>,<S>
    """

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))

    print_rich(pretty_repr(LR0ParsingTable(cfg)))
