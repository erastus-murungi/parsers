from functools import cache
from typing import Callable, NamedTuple

from typeguard import typechecked

from grammar import Grammar, NonTerminal
from ll.core import Part, TerminalSequenceSet, gen_parts
from ll.first_k import first_k
from utils.fixpoint import fixpoint, reduce


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

    @typechecked
    def append_transfer_function(
        transfer_function: TransferFunction, part: Part
    ) -> TransferFunction:
        if isinstance(part, TerminalSequenceSet):
            return lambda follow_set_unt: transfer_function(follow_set_unt).k_concat(
                part
            )
        else:
            return lambda follow_set_unt: transfer_function(follow_set_unt).k_concat(
                first_set[part]
            )

    first_set = first_k(grammar, k)

    equation_system: EquationSystem = {}
    for origin, expansion in grammar.iter_productions():
        parts = gen_parts(expansion, k)
        for part_index, part in enumerate(parts):
            if isinstance(part, NonTerminal):
                equation_system[
                    UniqueNonTerminal(origin, part, part_index)
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
