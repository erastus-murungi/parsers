from collections import defaultdict
from functools import cache
from typing import Callable, cast

from more_itertools import split_at

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalSequence, TerminalSequenceSet
from utils.fixpoint import fixpoint

MAX_ITERATIONS = 1000

FirstSet = dict[NonTerminal | Expansion, TerminalSequenceSet]
TransferFunction = Callable[[FirstSet], TerminalSequenceSet]
EquationSystem = dict[Expansion, TransferFunction]
ResultFunction = Callable[[FirstSet], TerminalSequenceSet]


def get_init_result_function(k: int) -> ResultFunction:
    return lambda result_vector: TerminalSequenceSet.eps(k)


def get_terminal_result_function(
    result_function: ResultFunction, terminals: tuple[Terminal, ...], k: int
) -> ResultFunction:
    terminal_strings = TerminalSequenceSet.of(TerminalSequence(terminals, k), k)
    return lambda result_vector: result_function(result_vector).k_concat(
        terminal_strings, k
    )


def get_non_terminal_result_function(
    result_function: ResultFunction, non_terminal: NonTerminal, k: int
) -> ResultFunction:
    def f(result_vector: FirstSet) -> TerminalSequenceSet:
        ts_set = result_vector[non_terminal]
        return result_function(result_vector).k_concat(ts_set, k)

    return f


def get_transfer_function(expansion: Expansion, k: int) -> TransferFunction:
    result_function = get_init_result_function(k)
    for symbols in split_at(
        expansion,
        pred=lambda symbol: isinstance(symbol, NonTerminal),
        keep_separator=True,
    ):
        if not symbols:
            continue
        if isinstance(symbols[0], Terminal):
            terminals = cast(tuple[Terminal, ...], tuple(symbols))
            result_function = get_terminal_result_function(
                result_function, terminals, k
            )
        else:
            (non_terminal,) = symbols
            result_function = get_non_terminal_result_function(
                result_function, non_terminal, k
            )

    return result_function


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
    equation_system: EquationSystem = {
        expansion: get_transfer_function(expansion, k)
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
