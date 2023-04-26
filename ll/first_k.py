from collections import defaultdict
from functools import cache
from typing import Callable, cast

from more_itertools import split_at
from rich.traceback import install

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalsSequence, TerminalStrings

install()


FirstSet = dict[NonTerminal | Expansion, TerminalStrings]
TransferFunction = Callable[[FirstSet], TerminalStrings]
EquationSystem = dict[Expansion, TransferFunction]
ResultFunction = Callable[[FirstSet], TerminalStrings]


def get_init_result_function(k: int) -> ResultFunction:
    return lambda result_vector: TerminalStrings.eps(k)


def get_terminal_result_function(
    result_function: ResultFunction, terminals: tuple[Terminal, ...], k: int
) -> ResultFunction:
    terminal_strings = TerminalStrings.of(TerminalsSequence(terminals, k), k)
    return lambda result_vector: result_function(result_vector).k_concat(
        terminal_strings, k
    )


def get_non_terminal_result_function(
    result_function: ResultFunction, non_terminal: NonTerminal, k: int
) -> ResultFunction:
    def f(result_vector: FirstSet) -> TerminalStrings:
        r = result_vector[non_terminal]
        return result_function(result_vector).k_concat(r, k)

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


def get_step_function(
    k: int, grammar: Grammar
) -> Callable[[EquationSystem, FirstSet], FirstSet]:
    def step_function(
        equation_system: EquationSystem, result_vector: FirstSet
    ) -> FirstSet:
        new_result_vector: FirstSet = defaultdict(lambda: TerminalStrings.empty(k))
        for origin, expansion in grammar.iter_productions():
            r = equation_system[expansion](result_vector)
            new_result_vector[expansion] = r
            new_result_vector[origin] |= r
        return new_result_vector

    return step_function


@cache
def first_k(grammar: Grammar, k: int) -> FirstSet:
    equation_system: EquationSystem = {
        expansion: get_transfer_function(expansion, k)
        for _, expansion in grammar.iter_productions()
    }

    result_vector: FirstSet
    if k <= 1:
        result_vector = {}
        for origin, expansions in grammar.items():
            result_vector[origin] = TerminalStrings.eps(k)
            for expansion in expansions:
                result_vector[expansion] = TerminalStrings.empty(k)

    else:
        result_vector = {
            lhs: rhs.increment_k(k) for lhs, rhs in first_k(grammar, k - 1).items()
        }

    step_function = get_step_function(k, grammar)

    iterations = 0
    while True:
        new_result_vector = step_function(equation_system, result_vector)
        if new_result_vector == result_vector:
            break
        result_vector = new_result_vector
        iterations += 1

    return dict(result_vector)


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR_LL5

    g = Grammar.from_str(*GRAMMAR_LL5)
    rich_print(pretty_repr(first_k(g, 5)))
