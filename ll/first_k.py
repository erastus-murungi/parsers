from collections import defaultdict
from functools import cache, reduce
from typing import Callable

from more_itertools import split_at

from grammar import Expansion, Grammar, NonTerminal, Terminal
from ll.core import TerminalSequence, TerminalSequenceSet
from utils.fixpoint import fixpoint

FirstSet = dict[NonTerminal | Expansion, TerminalSequenceSet]

# compute the first_k set of a terminal sequence or a non-terminal
# k, and the terminal sequence or non-terminal, and transfer_function of first element
# are bound to its scope
TransferFunction = Callable[[FirstSet], TerminalSequenceSet]
EquationSystem = dict[Expansion, TransferFunction]


def init_first_set(grammar: Grammar, k: int) -> FirstSet:
    if k <= 1:
        initial_first_set = {}
        for origin, expansions in grammar.items():
            initial_first_set[origin] = TerminalSequenceSet.eps(k)
            for expansion in expansions:
                initial_first_set[expansion] = TerminalSequenceSet.empty(k)
        return initial_first_set
    else:
        return {
            lhs: ts_set.increment_k(k)
            for lhs, ts_set in first_k(grammar, k - 1).items()
        }


@cache
def first_k(grammar: Grammar, k: int) -> FirstSet:
    def append_transfer_function(
        transfer_function: TransferFunction, symbols: list[Terminal] | list[NonTerminal]
    ) -> TransferFunction:
        match symbols:
            case [NonTerminal() as nt]:
                return lambda first_set: transfer_function(first_set).k_concat(
                    first_set[nt]
                )
            case [Terminal(), *_]:
                return lambda first_set: transfer_function(first_set).k_concat(
                    TerminalSequenceSet.of(TerminalSequence(symbols, k), k)
                )
            case []:
                return transfer_function

    equation_system: EquationSystem = {
        expansion: reduce(
            append_transfer_function,
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

    from utils.grammars import GRAMMAR_LL5

    g = Grammar.from_str(GRAMMAR_LL5)
    rich_print(pretty_repr(first_k(g, 2)))
