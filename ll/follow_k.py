from collections import ChainMap
from functools import cache
from typing import Callable, NamedTuple, cast

from more_itertools import split_at

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalsSequence, TerminalStrings
from ll.first_k import FirstSet, first_k


class UniqueNonTerminalIdentifier(NamedTuple):
    """
    We identify a non-terminal in a production rule by its
        1) origin,
        2) label
        3) position in the rule.
    """

    origin: NonTerminal
    non_terminal: NonTerminal
    position: int


ResultMap = dict[UniqueNonTerminalIdentifier, TerminalStrings]
FollowSet = dict[NonTerminal, TerminalStrings]
TransferFunction = Callable[[ResultMap, FollowSet], TerminalStrings]
EquationSystem = dict[UniqueNonTerminalIdentifier, TransferFunction]
ResultFunction = Callable[[ResultMap, FollowSet], TerminalStrings]


def get_step_function() -> Callable[[EquationSystem, ResultMap, FollowSet], ResultMap]:
    def step_function(
        equation_system: EquationSystem, result_map: ResultMap, follow_set: FollowSet
    ) -> ResultMap:
        new_result_map: ResultMap = {}
        for pos, _ in result_map.items():
            r = equation_system[pos](result_map, follow_set)
            follow_set[pos.non_terminal] |= r
            new_result_map[pos] = r
        return new_result_map

    return step_function


def get_init_result_function(k: int) -> ResultFunction:
    return lambda result_map, follow_set: TerminalStrings.eps(k)


def get_terminal_result_function(
    result_function: ResultFunction, terminals: tuple[Terminal, ...], k: int
) -> ResultFunction:
    terminal_strings = TerminalStrings.of(TerminalsSequence(terminals, k), k)
    return lambda result_map, follow_set: result_function(
        result_map, follow_set
    ).k_concat(terminal_strings, k)


def get_non_terminal_result_function(
    result_function: ResultFunction,
    non_terminal: NonTerminal,
    k: int,
    first_set: FirstSet,
) -> ResultFunction:
    first_nt = first_set[non_terminal]

    def updated_result_function(
        result_map: ResultMap, follow_set: FollowSet
    ) -> TerminalStrings:
        return result_function(result_map, follow_set).k_concat(first_nt, k)

    return updated_result_function


def get_combining_result_function(
    result_function: ResultFunction, k: int, origin
) -> ResultFunction:
    def combining_result_function(result_map: ResultMap, follow_set: FollowSet):
        r = follow_set[origin]
        return result_function(result_map, follow_set).k_concat(r, k)

    return combining_result_function


def get_partial_equation_system(
    origin: NonTerminal, expansion: Expansion, k: int, first_set: FirstSet
) -> EquationSystem:
    equation_system: EquationSystem = {}
    parts = [
        symbols
        for symbols in split_at(
            expansion,
            pred=lambda symbol: isinstance(symbol, NonTerminal),
            keep_separator=True,
        )
        if symbols
    ]
    nt_index = 0
    for part_index, syms in enumerate(parts):
        nt_index += part_index * len(syms)
        if isinstance(syms[0], NonTerminal):
            assert len(syms) == 1
            (sym,) = cast(tuple[NonTerminal], syms)
            result_function = get_init_result_function(k)
            for symbols in parts[part_index + 1 :]:
                if isinstance(symbols[0], Terminal):
                    terminals = cast(tuple[Terminal, ...], tuple(symbols))
                    result_function = get_terminal_result_function(
                        result_function, terminals, k
                    )
                else:
                    (non_terminal,) = cast(tuple[NonTerminal], tuple(symbols))
                    result_function = get_non_terminal_result_function(
                        result_function, non_terminal, k, first_set
                    )
            equation_system[
                UniqueNonTerminalIdentifier(origin, sym, nt_index)
            ] = get_combining_result_function(result_function, k, origin)
    return equation_system


@cache
def follow_k(grammar: Grammar, k: int) -> tuple[ResultMap, FollowSet]:
    assert k >= 1, "k must be greater than or equal to 1"

    first_set = first_k(grammar, k)

    equation_system: EquationSystem = dict(
        ChainMap(
            *(
                get_partial_equation_system(origin, expansion, k, first_set)
                for origin, expansion in grammar.iter_productions()
            )
        )
    )

    follow_set: FollowSet = {
        non_terminal: TerminalStrings.empty(k) for non_terminal in grammar.non_terminals
    }
    follow_set[grammar.start] = TerminalStrings.eof(k)

    result_map: ResultMap
    if k <= 1:
        result_map = {pos: TerminalStrings.empty(k) for pos in equation_system.keys()}
    else:
        prev = follow_k(grammar, k - 1)[0]
        result_map = {key: rhs.increment_k(k) for key, rhs in prev.items()}

    step_function = get_step_function()

    iterations = 1
    while True:
        new_result_map = step_function(equation_system, result_map, follow_set)
        if new_result_map == result_map:
            break
        result_map = new_result_map
        iterations += 1

    return result_map, follow_set


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR1

    g = Grammar.from_str(*GRAMMAR1)
    rich_print(pretty_repr(g))
    rich_print(pretty_repr(follow_k(g, 2)[1]))
