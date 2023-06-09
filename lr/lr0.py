from dataclasses import dataclass
from functools import cache

from more_itertools import one
from rich.traceback import install

from grammar import EMPTY, EOF, Expansion, NonTerminal, Symbol, Terminal
from lr.core import Accept, Goto, LRState, LRTable, Reduce, Shift

install(show_locals=True)


@dataclass(frozen=True, slots=True, eq=True)
class LR0Item:
    name: NonTerminal
    dot: int
    expansion: Expansion

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.expansion[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.expansion[self.dot:])}"
        )

    def __str__(self):
        return (
            f"{self.name!s} -> {' '.join(str(sym) for sym in self.expansion[:self.dot])}"
            f" . "
            f"{' '.join(str(sym) for sym in self.expansion[self.dot:])}"
        )

    def __iter__(self):
        yield from [self.name, self.dot, self.expansion]

    def advance(self):
        return LR0Item(self.name, self.dot + 1, self.expansion)

    def completed(self):
        return self.dot >= len(self.expansion)

    @property
    def at_start(self):
        return self.dot == 0


class LR0ParsingTable(LRTable[LR0Item]):
    def __init__(
        self,
        grammar,
        *,
        reduce: bool = True,
    ):
        """
        :param grammar: a context-free grammar
        :param reduce: if True, reduce actions are generated
        """
        super().__init__(grammar, reduce=reduce)

    def is_kernel_item(self, item: LR0Item) -> bool:
        """
        Return True if the item is a kernel item, False otherwise

        :param item: an LR(0) item
        :return: True if the item is a kernel item, False otherwise

        Kernel items : the initial item, (S' -> .S), and all items whose dots are at the left end.
        Kernel items are used to generate closure items.
        """

        return (item.name == self.grammar.start and item.at_start) or (
            not item.at_start
        )

    @cache
    def closure(self, kernel: LRState[LR0Item]):
        """
        Compute the closure of LR(0) item set

        :param kernel: a set of LR(0) items
              Kernel items : the initial item, (S' -> .S), and all items whose dots are at the left end.
        :return: closure of the set

        CLOSURE(I) = I ∪ { X → .γ | A → α • Xβ ∈ I, X → γ ∈ G }

        Algorithm:
            Closure(I) =
                repeat
                    for any item A → α.Xβ in I
                        for any production X → γ
                            I ← I ∪ { X → .γ }
                until I does not change.
                return I

        """

        # every item in L is in CLOSURE(L);
        closure: LRState[LR0Item] = kernel.copy()

        while True:
            initial_closure_size = len(closure)
            # a completed item is not in the closure
            for _, dot, rule in closure.yield_unfinished():
                b = rule[dot]  # α • Bβ
                if isinstance(b, NonTerminal):
                    # add all B → .γ to the closure
                    closure.extend(LR0Item(b, 0, gamma) for gamma in self.grammar[b])
            if len(closure) == initial_closure_size:
                break
        return closure

    @cache
    def goto(self, state: LRState[LR0Item], sym: Symbol) -> LRState[LR0Item]:
        """
        :param state: a state of the LR(0) state
        :param sym: a symbol
        :return: the state that results from shifting the dot over the symbol

        Goto is the transition function of the LR(0) automaton.
        It takes a state and a symbol and returns the state that
        results from shifting the dot over the symbol.
        """

        assert sym is not EMPTY and sym is not EOF
        kernel: LRState[LR0Item] = LRState(cls=state.type)
        for item in state.yield_unfinished():
            if item.expansion[item.dot] == sym:
                kernel.append(item.advance())
        return self.closure(kernel)

    @cache
    def init_kernel(self):
        return LRState[LR0Item](
            LR0Item(
                self.grammar.start,
                0,
                one(self.grammar[self.grammar.start]).append(EOF),
            ),
            cls=LR0Item,
        )

    def compute_reduce_actions(self):
        for state in self.states:
            for item in state.yield_finished():
                for symbol in self.grammar.terminals:
                    if (state, symbol.name) not in self:
                        self[(state, symbol.name)] = Reduce(
                            item.name, len(item.expansion)
                        )
                    else:
                        raise ValueError(
                            f"Encountered conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.name}\n"
                            f"  {self[(state, symbol.name)]} and \n"
                            f"  Reduce({item.name}, {len(item.expansion)})"
                        )

    def construct(self):
        # we use a dictionary to maintain states creation order
        states = {self.closure(self.init_kernel()): None}
        changing = True

        while changing:
            changing = False
            for state in list(states.keys()):
                for item in state.yield_unfinished():
                    dot, rule = item.dot, item.expansion
                    symbol = rule[dot]
                    if symbol is EOF:
                        # accept action
                        self[(state, symbol.name)] = Accept()
                        self.accept = (state, symbol.name)
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
                        if action != self.get((state, symbol.name), None):
                            self[(state, symbol.name)] = action
                            changing = True

        self.states = list(states)
        if self.reduce:
            self.compute_reduce_actions()
