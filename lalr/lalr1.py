from collections import defaultdict

from more_itertools import one

from grammar import CFG
from grammar.core import NonTerminal, Rule, Terminal
from lr.core import Goto, LRState, Reduce, Shift, T
from lr.lr0 import LR0Item, LR0ParsingTable


class AugmentedSymbol(NonTerminal):
    def __init__(self, symbol: NonTerminal, start: LRState[T], end: LRState[T]):
        super().__init__(f"{symbol!r} <{start.id!r}-{end.id!r}>")
        self.symbol: NonTerminal = symbol
        self.start: LRState[T] = start
        self.end: LRState[T] = end

    def __repr__(self):
        return f"{self.symbol!r} [{self.start.id!r}:{self.end.id!r}]"


class LALR1ParsingTable(LR0ParsingTable):
    def __init__(self, grammar: CFG):
        super().__init__(grammar, reduce=False)
        self.lookaheads = self.compute_lookaheads()
        self.compute_reduce_actions()

    def compute_reduce_actions(self):
        for state in self.states:
            for item in state.yield_finished():
                lookahead = self.lookaheads[state][item]
                for symbol in lookahead:
                    if (state, symbol.id) not in self:
                        self[(state, symbol.id)] = Reduce(item.name, len(item.rule))
                    else:
                        raise ValueError(
                            f"Encountered conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.id}\n"
                            f"  {self[(state, symbol.id)]} and \n"
                            f"  {Reduce(item.name, len(item.rule))}"
                        )

    def compute_augmented_grammar(
        self,
    ) -> tuple[CFG, dict[LRState, dict[NonTerminal, AugmentedSymbol]]]:
        augmented_grammar = CFG(
            start_symbol=AugmentedSymbol(
                self.grammar.start_symbol, self.states[0], LRState[LR0Item](cls=LR0Item)
            )
        )

        old2new: dict[LRState, dict[NonTerminal, AugmentedSymbol]] = defaultdict(dict)

        for start_state in self.states:
            for item in start_state:
                if item.at_start:
                    (name, dot, rule) = item
                    # trace out the path of this rule
                    augmented_rule = Rule()
                    new_name = AugmentedSymbol(
                        name, start_state, self.goto(start_state, name)
                    )
                    current_state = start_state
                    for symbol in rule:
                        if (current_state, symbol.id) == self.accept:
                            continue
                        match self[(current_state, symbol.id)]:
                            case Goto(next_state) | Shift(next_state):
                                if isinstance(symbol, NonTerminal):
                                    augmented_rule.append(
                                        AugmentedSymbol(
                                            symbol, current_state, next_state
                                        )
                                    )
                                else:
                                    augmented_rule.append(symbol)
                                current_state = next_state
                            case _:
                                raise Exception("Unexpected action")
                    old2new[start_state][name] = new_name
                    augmented_grammar.add_rule(new_name, augmented_rule)

        return augmented_grammar, old2new

    def get_only_completable_item(
        self, current_state: LRState[LR0Item], current_item: LR0Item
    ) -> LR0Item:
        return one(
            (
                item
                for item in current_state
                if item.name == current_item.name
                and item.rule == current_item.rule
                and (
                    item.completed()
                    or (item.rule[item.dot] and item.name == self.grammar.start_symbol)
                )
            ),
            too_short=ValueError(f"No completable item found in: {current_state}"),
            too_long=ValueError(f"Multiple completable items found in {current_state}"),
        )

    def compute_lookaheads(self) -> dict[LRState, dict[LR0Item, set[Terminal]]]:
        augmented_grammar, old2new = self.compute_augmented_grammar()
        # print_rich(pretty_repr(augmented_grammar))
        # print_rich(pretty_repr(State.ids))
        # print_rich(pretty_repr(augmented_grammar.follow()))

        augmented_follow = augmented_grammar.follow()
        lookaheads: dict[LRState, dict[LR0Item, set[Terminal]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # propagate the follow set
        for start_state in self.states:
            for current_item in start_state:
                if current_item.at_start:
                    current_state = start_state
                    for symbol in current_item.rule:
                        if (current_state, symbol.id) == self.accept:
                            continue
                        match self[(current_state, symbol.id)]:
                            case Goto(next_state) | Shift(next_state):
                                current_state = next_state
                            case _:
                                raise Exception("Unexpected action")

                    completed_item = self.get_only_completable_item(
                        current_state, current_item
                    )

                    augmented_name = old2new[start_state][completed_item.name]
                    lookaheads[current_state][completed_item] |= augmented_follow[
                        augmented_name
                    ]

        # print_rich(pretty_repr(lookaheads))
        return lookaheads


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr

    from utils.parse_grammar import parse_grammar

    table = {
        "+": "+",
        ";": ";",
        "(": "(",
        ")": ")",
        "=": "=",
        "*": "*",
    }

    g = """
        <S>
        <S> -> <E>
        <E> -> <L> = <R> | <R>
        <L> -> char | *<R>
        <R> -> <L>
    """

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))
    p = LALR1ParsingTable(cfg)
    p.draw_with_graphviz()
    print_rich(pretty_repr(p))
