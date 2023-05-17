# this module detects Left Recursion in a grammar, both direct and indirect.
from collections import defaultdict
from typing import Iterable

from grammar import Grammar, NonTerminal


def compute_left_recursion_non_terminals(grammar: Grammar) -> Iterable[NonTerminal]:
    """Compute the set of non-terminals that have left recursion in the grammar.
    :param grammar: a grammar
    :return: the set of non-terminals that have left recursion in the grammar
    """

    nullables = grammar.gen_nullable()
    starters: dict[NonTerminal, set[NonTerminal]] = defaultdict(set)

    # For all non-terminals A, B calculate the relation 'A can-start-with B'
    for origin, expansion in grammar.iter_productions():
        for symbol in expansion:
            if isinstance(symbol, NonTerminal):
                starters[origin].add(symbol)
                if symbol not in nullables:
                    break
            else:
                break

    # Calculate transitive closure of the relation 'A can-start-with B'
    #  Ex.: A->B, B->C => A->{B, C}
    changed = True
    while changed:
        changed = False
        for A in tuple(starters.keys()):
            entries_copy = starters[A].copy()
            for B in entries_copy:
                starters[A].update(starters[B])
                if entries_copy < starters[A]:
                    changed = True

    # If A can-start-with A, then the grammar is left-recursive
    for A, entries in starters.items():
        if A in entries:
            yield A


def has_left_recursion(grammar: Grammar) -> bool:
    """
    Detect if a grammar has left recursion.
    :param grammar: a grammar
    :return: True if the grammar has left recursion, False otherwise
    """
    return any(compute_left_recursion_non_terminals(grammar))
