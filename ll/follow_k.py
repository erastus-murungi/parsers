from functools import cache, reduce
from typing import Callable, NamedTuple

from more_itertools import split_at

from grammar import Grammar, NonTerminal, Terminal
from ll.core import TerminalSequence, TerminalSequenceSet
from ll.first_k import first_k
from utils.fixpoint import fixpoint


class UniqueNonTerminal(NamedTuple):
    """
    We identify a non-terminal in a production rule by its
        1) origin,
        2) label
        3) position in the rule.
    """

    origin: NonTerminal
    non_terminal: NonTerminal
    part_index: int


FollowSetUNT = dict[UniqueNonTerminal, TerminalSequenceSet]
FollowSet = dict[NonTerminal, TerminalSequenceSet]
TransferFunction = Callable[[FollowSetUNT], TerminalSequenceSet]
EquationSystem = dict[UniqueNonTerminal, TransferFunction]


@cache
def follow_k(grammar: Grammar, k: int) -> tuple[FollowSetUNT, FollowSet]:
    assert k >= 1, "k must be greater than or equal to 1"

    def merge_transfer_functions(
        transfer_function: TransferFunction, non_terminal: NonTerminal
    ) -> TransferFunction:
        def _merge_results(follow_set_unt: FollowSetUNT) -> TerminalSequenceSet:
            follow_set_origin = follow_set[non_terminal]
            return transfer_function(follow_set_unt).k_concat(follow_set_origin)

        return _merge_results

    def append_transfer_function(
        transfer_function: TransferFunction, symbols: list[Terminal] | list[NonTerminal]
    ) -> TransferFunction:
        match symbols:
            case [NonTerminal() as non_terminal]:
                return lambda follow_set_unt: transfer_function(
                    follow_set_unt
                ).k_concat(first_set[non_terminal])
            case [Terminal(), *_]:
                return lambda follow_set_unt: transfer_function(
                    follow_set_unt
                ).k_concat(TerminalSequenceSet.of(TerminalSequence(symbols, k), k))
            case []:
                return transfer_function
            case _:
                raise ValueError(
                    "symbols must be a list of terminals or a single non-terminal"
                )

    first_set = first_k(grammar, k)

    equation_system: EquationSystem = {}
    for origin, expansion in grammar.iter_productions():
        parts = tuple(
            split_at(
                expansion,
                pred=lambda symbol: isinstance(symbol, NonTerminal),
                keep_separator=True,
            )
        )

        for part_index, syms in enumerate(parts):
            match syms:
                case [NonTerminal() as nt]:
                    equation_system[
                        UniqueNonTerminal(origin, nt, part_index)
                    ] = merge_transfer_functions(
                        reduce(
                            append_transfer_function,
                            parts[part_index + 1 :],
                            lambda _: TerminalSequenceSet.eps(k),
                        ),
                        origin,
                    )

    follow_set: FollowSet = {
        non_terminal: TerminalSequenceSet.empty(k)
        for non_terminal in grammar.non_terminals
    }
    follow_set[grammar.start] = TerminalSequenceSet.eof(k)

    @fixpoint
    def step_function(follow_set_unt: FollowSetUNT) -> FollowSetUNT:
        new_follow_set_unt: FollowSetUNT = {}
        for unique_nt, _ in follow_set_unt.items():
            ts_set = equation_system[unique_nt](follow_set_unt)
            follow_set[unique_nt.non_terminal] |= ts_set
            new_follow_set_unt[unique_nt] = ts_set
        return new_follow_set_unt

    initial_follow_set_unt = (
        {unique_nt: TerminalSequenceSet.empty(k) for unique_nt in equation_system}
        if k <= 1
        else {
            unique_nt: ts_set.increment_k(k)
            for unique_nt, ts_set in follow_k(grammar, k - 1)[0].items()
        }
    )

    return step_function(initial_follow_set_unt), follow_set


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR_JSON

    g = Grammar.from_str(GRAMMAR_JSON)
    rich_print(pretty_repr(g))
    rich_print(pretty_repr(follow_k(g, 2)[1]))
