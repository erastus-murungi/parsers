from collections import defaultdict
from functools import lru_cache
from typing import Sequence

from .core import (
    EMPTY,
    EOF,
    Definition,
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


class CFG(dict[NonTerminal, Definition]):
    """
    Notes: https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf
    """

    __slots__ = (
        "_start",
        "_terminals",
        "_current_time",
    )

    def __init__(self, start_symbol: NonTerminal):
        super().__init__()
        self._start: NonTerminal = start_symbol
        self._terminals: set[Terminal] = {EOF}
        self._current_time = 0

    def __hash__(self):
        unique_print = (id(self), self._current_time)
        return hash(unique_print)

    def __len__(self):
        return super().__len__() - 1

    @property
    def start(self) -> NonTerminal:
        return self._start

    @property
    def non_terminals(self) -> set[NonTerminal]:
        return set(self.keys())

    @property
    def terminals(self) -> set[Terminal]:
        return self._terminals.copy()

    def add_rule(self, origin: NonTerminal, expansion: Expansion) -> None:
        assert isinstance(origin, NonTerminal)
        if EOF in expansion:
            raise ValueError(
                "you are not allowed to explicit add an EOF token, "
                "it is implicitly added by the grammar object"
            )
        if EMPTY in expansion:
            raise ValueError(
                "you are not allowed to explicit add a sentinel, "
                "pass in empty SententialForm instead e.g "
                "`add_rule(var, Rule([]))`"
            )

        if origin == self._start:
            if origin in self and len(self[origin]) > 0:
                raise ValueError(
                    "you are not allowed to add a rule of the form "
                    f"`<START> => rule1 | rule2 | rule3` because it is ambiguous\n"
                    f"The start symbol should have only one production rule, implicitly by an EOF token"
                )
            else:
                super().__setitem__(origin, Definition([expansion]))
        else:
            self._terminals.update(
                (symbol for symbol in expansion if isinstance(symbol, Terminal))
            )
            if origin not in self:
                super().__setitem__(origin, Definition())
            if len(expansion) == 0:
                self[origin].append(expansion.append_marker(EMPTY))
            else:
                self[origin].append(expansion)
        self._current_time += 1

    def add_definition(self, origin: NonTerminal, definition: Definition) -> None:
        if origin in self:
            raise ValueError(
                "you are not allowed overwrite a definition that is already in the grammar"
            )
        self[origin] = definition
        self._current_time += 1

    def __repr__(self) -> str:
        return "\n".join(
            f"{repr(rhs)} => {repr(definition)}" for rhs, definition in self.items()
        )

    @lru_cache(maxsize=1)  # only remember the last nullable set
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

    @lru_cache(maxsize=1)  # only remember the last first set
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

    @lru_cache(maxsize=1)  # only remember the last follow set
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
