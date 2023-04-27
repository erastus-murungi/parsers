from collections import defaultdict
from functools import cache
from typing import Callable, cast

from more_itertools import split_at, first, one

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalSequence, TerminalSequenceSet
from utils.fixpoint import fixpoint
from functools import reduce

MAX_ITERATIONS = 1000

FirstSet = dict[NonTerminal | Expansion, TerminalSequenceSet]
TransferFunction = Callable[[FirstSet], TerminalSequenceSet]
EquationSystem = dict[Expansion, TransferFunction]
ResultFunction = Callable[[FirstSet], TerminalSequenceSet]


def init_first_set(grammar: Grammar, k: int) -> FirstSet:
    if k <= 1:
        initial_first_set = {}
        for origin, expansions in grammar.items():
            initial_first_set[origin] = TerminalSequenceSet.eps(k)
            for expansion in expansions:
                initial_first_set[expansion] = TerminalSequenceSet.empty(k)
        return initial_first_set
    else:
        return {lhs: rhs.increment_k(k) for lhs, rhs in first_k(grammar, k - 1).items()}


@cache
def first_k(grammar: Grammar, k: int) -> FirstSet:
    def concat(
        result_function: ResultFunction, symbols: list[Terminal] | list[NonTerminal]
    ) -> ResultFunction:
        match symbols:
            case [NonTerminal()]:
                return lambda first_set: result_function(first_set).k_concat(
                    first_set[one(symbols)], k
                )
            case [Terminal(), *_]:
                return lambda first_set: result_function(first_set).k_concat(
                    TerminalSequenceSet.of(TerminalSequence(symbols, k), k), k
                )
            case []:
                return result_function

    equation_system: EquationSystem = {
        expansion: reduce(
            concat,
            split_at(
                expansion,
                pred=lambda symbol: isinstance(symbol, NonTerminal),
                keep_separator=True,
            ),
            lambda _: TerminalSequenceSet.eps(k),
        )
        for _, expansion in grammar.iter_productions()
    }

    @fixpoint
    def step_function(first_set: FirstSet) -> FirstSet:
        updated_first_set: FirstSet = defaultdict(lambda: TerminalSequenceSet.empty(k))
        for origin, expansion in grammar.iter_productions():
            ts_set = equation_system[expansion](first_set)
            updated_first_set[expansion] = ts_set
            updated_first_set[origin] |= ts_set
        return updated_first_set

    return step_function(init_first_set(grammar, k))


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR_JSON

    g = Grammar.from_str(GRAMMAR_JSON)
    rich_print(pretty_repr(first_k(g, 2)))
