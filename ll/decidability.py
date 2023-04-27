from itertools import takewhile
from typing import Optional

from grammar import Grammar, NonTerminal
from ll.core import TerminalSequenceSet
from ll.first_k import first_k
from ll.follow_k import follow_k


def is_decidable(
    grammar: Grammar, non_terminal: NonTerminal, max_k: int
) -> Optional[int]:
    """
    For every pair of production rules A->α|β, the following condition holds.
        FIRST𝑘(αFOLLOW𝑘(A)) ∩ FIRST𝑘( β FOLLOW𝑘(A)) = ∅

    :param grammar: A grammar
    :param non_terminal: a non-terminal to decide how many lookahead symbols are needed
    :param max_k: the maximum number of lookahead symbols to consider

    :return: the number of lookahead symbols needed to decide the non-terminal if it is decidable, else None
    """

    assert max_k >= 1, "max_k must be at least 1"

    expansions = grammar[non_terminal]
    if not expansions:
        raise ValueError(f"Non-terminal {non_terminal} has no expansions")

    # only one expansion, so it is trivially decidable
    if len(expansions) == 1:
        return 0

    for k in range(1, max_k + 1):
        first_set = first_k(grammar, k)
        follow_set = follow_k(grammar, k)[1]

        follow_nt = follow_set[non_terminal]
        if not TerminalSequenceSet.inter(
            tuple(first_set[expansion].k_concat(follow_nt) for expansion in expansions),
            k,
        ):
            return k
    return None


def compute_k(grammar: Grammar, max_k: int = 10) -> Optional[int]:
    """
    Compute the maximum number of lookahead symbols needed to decide the grammar.

    :param grammar: the grammar to decide
    :param max_k: the maximum number of lookahead symbols to consider
    :return: the maximum number of lookahead symbols needed to decide the grammar if it is decidable, else None
    """
    return max(
        takewhile(
            lambda res: res is not None,
            (is_decidable(grammar, non_terminal, max_k) for non_terminal in grammar),
        ),
        default=None,
    )


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR_DECAF, RESERVED_DECAF

    g = Grammar.from_str(GRAMMAR_DECAF, RESERVED_DECAF, True)
    rich_print(pretty_repr(compute_k(g)))
