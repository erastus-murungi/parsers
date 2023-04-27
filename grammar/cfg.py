import re
from collections import defaultdict, deque
from itertools import count
from typing import Iterator, Optional, Sequence

from more_itertools import first, sliced

from utils.frozendict import FrozenDict

from .common import common_patterns
from .core import (
    DUMMY_LOC,
    EMPTY,
    EOF,
    Expansion,
    FirstSet,
    FollowSet,
    NonTerminal,
    NullableSet,
    Symbol,
    Terminal,
    Tokenizer,
)


def update_set(set1, set2):
    if not set2 or set1 > set2:
        return False

    copy = set(set1)
    set1 |= set2
    return set1 != copy


class Grammar(FrozenDict[NonTerminal, frozenset[Expansion]]):
    __slots__ = ("terminals", "non_terminals", "start", "orig_start", "tokenizer")

    def __init__(
        self,
        mapping: dict[NonTerminal, frozenset[Expansion]],
        terminals: frozenset[Terminal],
        start: NonTerminal,
        orig_start: NonTerminal,
        non_terminals: frozenset[NonTerminal],
        tokenizer: Tokenizer,
    ):
        super().__init__(mapping)
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.orig_start = orig_start
        self.start = start
        self.tokenizer = tokenizer

    def iter_productions(self) -> Iterator[tuple[NonTerminal, Expansion]]:
        for origin, expansions in self.items():
            for expansion in expansions:
                yield origin, expansion

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
                    for i, sym in expansion.enumerate_non_terminals():
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

    @staticmethod
    def from_str(
        grammar_str: str,
        reserved_words: frozenset[str] = frozenset(),
        transform_regex_to_right=False,
    ) -> "Grammar":
        return _parse_grammar(grammar_str, reserved_words, transform_regex_to_right)

    class Builder:
        """
        Notes: https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf
        """

        __slots__ = ("_implicit_start", "_dict", "_start")

        def __init__(
            self,
            start: Optional[NonTerminal] = None,
            implicit_start_name: str = "Start",
        ) -> None:
            super().__init__()
            self._dict: dict[NonTerminal, set[Expansion]] = defaultdict(set)
            self._implicit_start: NonTerminal = NonTerminal(implicit_start_name)
            self._start = start

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
            # it is always assumed that the first symbol of your grammar is the start symbol
            if origin == self._implicit_start:
                raise ValueError(
                    f"grammar with name {self._implicit_start} not allowed \n"
                    f"{self._implicit_start} is an implicit start symbol used by the grammar object \n"
                    f"you can change the name of the start symbol by "
                    f"passing in a different name to the grammar builder"
                )

            return self.add_expansion_no_check(origin, seq)

        def add_expansion_no_check(
            self, origin: NonTerminal, seq: Sequence[Symbol]
        ) -> "Grammar.Builder":
            self._dict[origin].add(Expansion(seq))
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

        def build(self, tokenizer: Tokenizer) -> "Grammar":
            if not self._dict:
                raise ValueError("grammar must have at least one rule")
            orig_start = self._start or first(self._dict)
            return Grammar(
                mapping={
                    **{self._implicit_start: frozenset([Expansion({orig_start})])},
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
                orig_start=orig_start,
                tokenizer=tokenizer,
            )

    def get_mutable_copy(self) -> dict[NonTerminal, list[Expansion]]:
        mutable_copy = defaultdict(list)
        for origin, expansions in self.items():
            mutable_copy[origin] = list(expansions)
        return mutable_copy


temps_counter = count(0)


def iter_symbol_tokens(input_str: str) -> Iterator[str]:
    input_str = input_str.strip()
    while input_str:
        if m := re.match(r"^\|", input_str):
            yield m.group(0)
        elif m := re.match(r"<\w+>[?*+]?", input_str):  # NonTerminal
            yield m.group(0)
        elif m := re.match(r"((?<!')\(.*\)(?!')[?*+]?)", input_str):  # Grouped items
            yield m.group(0)
        elif m := re.match(r"'\w+?'[?*+]?", input_str):  # 'any word literal'
            yield m.group(0)
        elif m := re.match(r"r'.*'", input_str):  # any number
            yield m.group(0)
        elif m := re.match(r"\w+", input_str):  # keyword
            yield m.group(0)
        elif m := re.match(r"'.*?'[?*+]?", input_str):  # any literal
            yield m.group(0)
        else:
            raise ValueError(f"Invalid token: {input_str}")
        input_str = input_str[m.end() :].strip()


def _parse_grammar(
    grammar_str: str,
    reserved_words: frozenset[str] = frozenset(),
    transform_regex_to_right_recursive: bool = False,
) -> Grammar:
    """Ad Hoc grammar parser"""
    grammar_builder = Grammar.Builder()
    patterns: dict[str, re.Pattern] = {}
    for origin_str, definition_str in sliced(
        re.split(r"<(\w+)>\s*->", grammar_str.strip())[1:], n=2, strict=True
    ):
        origin = NonTerminal(origin_str)
        queue = deque([(origin, definition_str)])
        while queue:
            origin, expansion = queue.popleft()
            if isinstance(expansion, str):
                rule: list[Symbol] = []
                for lexeme in iter_symbol_tokens(expansion):
                    if lexeme == "|":
                        queue.append((origin, rule if rule else (EMPTY,)))
                        rule = []
                    elif (
                        lexeme.endswith("?")
                        | lexeme.endswith("*")
                        | lexeme.endswith("+")
                    ):
                        if lexeme.startswith("("):
                            R = NonTerminal(f"R_{next(temps_counter)}")
                            queue.append((R, lexeme[1:-2]))
                        elif lexeme.startswith("'"):
                            R = NonTerminal(f"R_{next(temps_counter)}")
                            queue.append((R, lexeme[:-1]))
                        else:
                            assert lexeme.startswith("<")
                            R = NonTerminal(lexeme[1:-2])

                        N = NonTerminal(
                            f"N_{next(temps_counter)}", original_repr=lexeme
                        )
                        if lexeme.endswith("?"):
                            # R? ⇒    N → ε
                            #         N → R
                            queue.append((N, (EMPTY,)))
                            queue.append((N, (R,)))
                        elif lexeme.endswith("*"):
                            # R* ⇒    N → ε
                            queue.append((N, (EMPTY,)))
                            if transform_regex_to_right_recursive:
                                # N → R N
                                queue.append((N, (R, N)))
                            else:
                                # N → N R
                                queue.append((N, (N, R)))
                        else:
                            # R+ ⇒    N → R
                            assert lexeme.endswith("+")
                            queue.append((N, (R,)))
                            if transform_regex_to_right_recursive:
                                # N -> R N
                                queue.append((N, (R, N)))
                            else:
                                # N → N R
                                queue.append((N, (N, R)))
                        rule.append(N)

                    elif lexeme == "<>":
                        rule.append(EMPTY)
                    elif lexeme.startswith(r"\\"):
                        # this is a terminal
                        rule.append(Terminal(lexeme[1:], lexeme[1:], DUMMY_LOC))
                    elif lexeme.startswith("<"):
                        # this is a non-terminal
                        rule.append(NonTerminal(lexeme[1:-1]))
                    elif lexeme in common_patterns:
                        patterns[lexeme] = common_patterns[lexeme]
                        # keywords
                        rule.append(Terminal(lexeme, lexeme, DUMMY_LOC))
                    elif re.match(r"'.*'", lexeme):
                        lexeme = lexeme[1:-1]
                        patterns[lexeme] = re.compile(
                            re.escape(lexeme), flags=re.DOTALL
                        )
                        rule.append(
                            Terminal(
                                lexeme,
                                lexeme,
                                DUMMY_LOC,
                            )
                        )
                    elif re.match(r"r'.*'", lexeme):
                        lexeme = lexeme[2:-1]
                        patterns[lexeme] = re.compile(lexeme, flags=re.DOTALL)
                        rule.append(
                            Terminal(
                                lexeme,
                                lexeme,
                                DUMMY_LOC,
                            )
                        )
                    else:
                        raise ValueError(f"unknown symbol {lexeme}\n{expansion}")
                grammar_builder.add_expansion(origin, rule)
            else:
                grammar_builder.add_expansion(origin, expansion)

    return grammar_builder.build(Tokenizer(patterns, reserved_words))
