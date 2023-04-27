from collections import ChainMap
from functools import cache
from typing import Callable, NamedTuple, cast

from more_itertools import split_at

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalSequence, TerminalSequenceSet
from ll.first_k import FirstSet, first_k
from utils.fixpoint import fixpoint

MAX_ITERATIONS = 1000


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


ResultMap = dict[UniqueNonTerminalIdentifier, TerminalSequenceSet]
FollowSet = dict[NonTerminal, TerminalSequenceSet]
TransferFunction = Callable[[ResultMap, FollowSet], TerminalSequenceSet]
EquationSystem = dict[UniqueNonTerminalIdentifier, TransferFunction]
ResultFunction = Callable[[ResultMap, FollowSet], TerminalSequenceSet]


def get_init_result_function(k: int) -> ResultFunction:
    return lambda result_map, follow_set: TerminalSequenceSet.eps(k)


def get_terminal_result_function(
    result_function: ResultFunction, terminals: tuple[Terminal, ...], k: int
) -> ResultFunction:
    terminal_strings = TerminalSequenceSet.of(TerminalSequence(terminals, k), k)
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
    ) -> TerminalSequenceSet:
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


def get_init_result_map(grammar, k, equation_system) -> ResultMap:
    if k <= 1:
        return {pos: TerminalSequenceSet.empty(k) for pos in equation_system.keys()}
    else:
        prev = follow_k(grammar, k - 1)[0]
        return {key: rhs.increment_k(k) for key, rhs in prev.items()}


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
        non_terminal: TerminalSequenceSet.empty(k)
        for non_terminal in grammar.non_terminals
    }
    follow_set[grammar.start] = TerminalSequenceSet.eof(k)

    @fixpoint
    def step_function(result_map: ResultMap) -> ResultMap:
        new_result_map: ResultMap = {}
        for pos, _ in result_map.items():
            ts_set = equation_system[pos](result_map, follow_set)
            follow_set[pos.non_terminal] |= ts_set
            new_result_map[pos] = ts_set
        return new_result_map

    return step_function(get_init_result_map(grammar, k, equation_system)), follow_set


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR1

    g = Grammar.from_str(*GRAMMAR1)
    rich_print(pretty_repr(g))
    rich_print(pretty_repr(follow_k(g, 2)[1]))
