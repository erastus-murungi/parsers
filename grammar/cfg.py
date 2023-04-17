from collections import defaultdict
from typing import Sequence

from more_itertools import first

from utils.frozendict import FrozenDict

from .core import (
    EMPTY,
    EOF,
    Expansion,
    FirstSet,
    FollowSet,
    NonTerminal,
    NullableSet,
    Symbol,
    Terminal,
)


def update_set(set1, set2):
    if not set2 or set1 > set2:
        return False

    copy = set(set1)
    set1 |= set2
    return set1 != copy


class Grammar(FrozenDict[NonTerminal, frozenset[Expansion]]):
    __slots__ = ("terminals", "non_terminals", "start")

    def __init__(
        self,
        mapping: dict[NonTerminal, frozenset[Expansion]],
        terminals: frozenset[Terminal],
        start: NonTerminal,
        non_terminals: frozenset[NonTerminal],
    ):
        super().__init__(mapping)
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.start = start

    def gen_nullable(self) -> NullableSet:
        NULLABLE: NullableSet = {EMPTY}

        changed = True
        while changed:
            changed = False
            for origin, expansions in self.items():
                for expansion in expansions:
                    if set(expansion) <= NULLABLE:
                        if update_set(NULLABLE, {origin}):
                            changed = True
                            break  # move on to the next origin
        return NULLABLE

    def first(self, symbols: Sequence[Symbol]) -> set[Terminal]:
        if not symbols:
            return {EMPTY}

        x, *xs = symbols
        FIRST = self.gen_first()
        return FIRST[x] | self.first(xs) if (x in self.gen_nullable()) else FIRST[x]

    def gen_first(self) -> FirstSet:
        FIRST: FirstSet = defaultdict(set)
        FIRST.update({terminal: {terminal} for terminal in self.terminals})

        changed = True
        nullable = self.gen_nullable()
        while changed:
            changed = False
            for origin, expansions in self.items():
                for expansion in expansions:
                    for i, sym in enumerate(expansion):
                        if set(expansion[:i]) <= nullable:
                            if update_set(FIRST[origin], FIRST[sym]):
                                changed = True
                        else:
                            break
        return FIRST

    def gen_follow(self) -> FollowSet:
        FOLLOW: FollowSet = defaultdict(set)
        FOLLOW[self.start] = {EOF}

        changed = True
        while changed:
            changed = False
            for origin, expansions in self.items():
                for expansion in expansions:
                    for i, sym in expansion.enumerate_variables():
                        successor = self.first(expansion[i + 1 :])
                        if EMPTY in successor and update_set(
                            FOLLOW[sym], FOLLOW[origin]
                        ):
                            changed = True
                        if update_set(FOLLOW[sym], successor - {EMPTY}):
                            changed = True

        return FOLLOW

    def __setitem__(self, key, value):
        raise Exception("Cannot modify grammar; use add_rule instead")

    def __str__(self) -> str:
        return "\n".join(
            f"{rhs!s} => {expansion!s}"
            for rhs, definition in self.items()
            for expansion in definition
        )

    def __repr__(self) -> str:
        return "\n".join(
            f"{rhs!r} => {expansion!r}"
            for rhs, definition in self.items()
            for expansion in definition
        )

    class Builder:
        """
        Notes: https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf
        """

        __slots__ = ("_implicit_start", "_dict")

        def __init__(self, start="START") -> None:
            super().__init__()
            self._dict: dict[NonTerminal, set[Expansion]] = defaultdict(set)
            self._implicit_start: NonTerminal = NonTerminal(start.capitalize())

        def add_expansion(
            self, origin: NonTerminal, seq: Sequence[Symbol]
        ) -> "Grammar.Builder":
            assert isinstance(origin, NonTerminal)
            if EOF in seq:
                raise ValueError(
                    "you are not allowed to explicit add an EOF token, "
                    "it is implicitly added by the grammar object"
                )
            if seq.count(EMPTY) > 1:
                raise ValueError(
                    "you cannot have more than one empty symbol in an expansion"
                )
            expansion = Expansion(seq)

            # it is always assumed that the first symbol of your grammar is the start symbol
            if origin == self._implicit_start:
                raise ValueError(
                    f"grammar with name {self._implicit_start} not allowed \n"
                    f"{self._implicit_start} is an implicit start symbol used by the grammar object \n"
                    f"you can change the name of the start symbol by "
                    f"passing in a different name to the grammar builder"
                )
            self._dict[origin].add(expansion)
            return self

        def add_definition(
            self, origin: NonTerminal, definition: set[Expansion]
        ) -> "Grammar.Builder":
            if origin in self._dict:
                raise ValueError(
                    "you are not allowed overwrite a definition that is already in the grammar"
                )
            self._dict[origin] = definition
            return self

        def build(self) -> "Grammar":
            if not self._dict:
                raise ValueError("grammar must have at least one rule")
            return Grammar(
                mapping={
                    **{
                        self._implicit_start: frozenset(
                            [Expansion({first(self._dict)})]
                        )
                    },
                    **{
                        origin: frozenset(expansions)
                        for origin, expansions in self._dict.items()
                    },
                },
                terminals=frozenset(
                    {EOF}
                    | {
                        symbol
                        for expansions in self._dict.values()
                        for expansion in expansions
                        for symbol in expansion
                        if isinstance(symbol, Terminal)
                    }
                ),
                start=self._implicit_start,
                non_terminals=frozenset(self._dict.keys()),
            )
