from collections import defaultdict, deque
from functools import reduce
from typing import Optional, Sequence, cast

from core import (
    EMPTY,
    EOF,
    Definition,
    FirstSet,
    FollowSet,
    Node,
    NonTerminal,
    ParseTableEntry,
    ParseTreeSearchSpaceNode,
    ParsingTable,
    Rule,
    Symbol,
    Terminal,
)
from tokenizer import Tokenizer


class CFG(dict[NonTerminal, Definition]):
    __slots__ = (
        "_start_symbol",
        "_terminals",
        "_current_time",
        "_caches",
    )

    def __init__(self, start_symbol: NonTerminal):
        super().__init__()
        self._start_symbol: NonTerminal = start_symbol
        self._terminals: set[Terminal] = {EOF}
        self._current_time = 0
        self._caches: dict[
            str, tuple[int, set[Symbol] | FollowSet | FirstSet | ParsingTable]
        ] = {}

    def __len__(self):
        return super().__len__() - 1

    @property
    def non_terminals(self) -> set[NonTerminal]:
        return set(self.keys())

    @property
    def terminals(self) -> set[Terminal]:
        return self._terminals.copy()

    def add_rule(self, rhs: NonTerminal, rule: Rule) -> None:
        assert isinstance(rhs, NonTerminal)
        if EOF in rule:
            raise ValueError(
                "you are not allowed to explicit add an EOF token, "
                "it is implicitly added by the grammar object"
            )
        if EMPTY in rule:
            raise ValueError(
                "you are not allowed to explicit add a sentinel, "
                "pass in empty SententialForm instead e.g "
                "`add_rule(var, Rule([]))`"
            )

        if rhs == self._start_symbol:
            if rhs in self and len(self[rhs]) > 1:
                raise ValueError(
                    "you are not allowed to add a rule of the form "
                    f"`<START> => rule1 | rule2 | rule3` because it is ambiguous"
                    f"The start symbol should have only one production rule, terminated by an EOF token"
                )
            else:
                super().__setitem__(rhs, Definition([rule]))
        else:
            self._terminals.update(
                (symbol for symbol in rule if isinstance(symbol, Terminal))
            )
            if rhs not in self:
                super().__setitem__(rhs, Definition())
            if len(rule) == 0:
                self[rhs].append(rule.append_marker(EMPTY))
            else:
                self[rhs].append(rule)
        self._current_time += 1

    def add_definition(self, rhs: NonTerminal, definition: Definition) -> None:
        self[rhs] = definition

    def __repr__(self) -> str:
        return "\n".join(
            f"{repr(rhs)} => {repr(definition)}" for rhs, definition in self.items()
        )

    def leftmost_top_down_parsing_bfs(
        self, tokens: list[Tokenizer.Token], max_iters: int = 1000_000
    ):
        """
        Enormous time and memory usage:
            ● Lots of wasted effort:
                – Generates a lot of sentential forms that couldn't
                    possibly match.
                – But in general, extremely hard to tell whether a
                    sentential replacement can match – that's the job of
                    parsing!
            ● High branching factor:
                – Each sentential replacement can expand in (potentially)
                many ways for each non-terminal it contains.
        """

        start = Rule((self._start_symbol,))
        root: Node = Node(start)
        rules: deque[ParseTreeSearchSpaceNode] = deque([((root,), start)])
        seen: set[Rule] = set()
        nullable_set: set[Symbol] = self.nullable()

        n_iters = 0
        while rules and n_iters < max_iters:
            tree, rule = rules.popleft()

            seen.add(rule)

            if rule.matches(tokens):
                return tree

            for index, symbol in rule.enumerate_variables():
                for replacement in self[symbol]:
                    if (
                        next_form := rule.perform_derivation(index, replacement)
                    ).should_prune(tokens, seen, nullable_set):
                        continue
                    rules.append(
                        (
                            tree[:-1]
                            + tree[-1].update(
                                index, ParseTableEntry(symbol, replacement), next_form
                            ),
                            next_form,
                        )
                    )

            n_iters += 1

    def leftmost_top_down_parsing_dfs(
        self, tokens: list[Tokenizer.Token], max_iters: int = 1000_000
    ):
        start = Rule((self._start_symbol,))
        root = Node(start)
        stack: list[ParseTreeSearchSpaceNode] = [((root,), start)]
        seen = set()
        nullable_set = self.nullable()

        n_iters = 0
        while stack and n_iters < max_iters:
            tree, rule = stack.pop()

            seen.add(rule)

            if rule.matches(tokens):
                return tree

            next_in_stack = []

            for index, symbol in rule.enumerate_variables():
                for replacement in self[symbol]:
                    if (
                        next_form := rule.perform_derivation(index, replacement)
                    ).should_prune(tokens, seen, nullable_set):
                        continue

                    next_in_stack.append(
                        (
                            tree[:-1]
                            + tree[-1].update(
                                index, ParseTableEntry(symbol, replacement), next_form
                            ),
                            next_form,
                        )
                    )
            stack.extend(reversed(next_in_stack))
            n_iters += 1

    def get_cached(
        self, function_name: str
    ) -> Optional[set[Symbol] | FollowSet | FirstSet | ParsingTable]:
        if function_name not in self._caches:
            return None
        compute_time, cached = self._caches[function_name]
        if compute_time == self._current_time:
            return cached
        return None

    def cache(self, function_name: str, data):
        self._caches[function_name] = (self._current_time, data)

    def is_nullable_sentential_form(
        self,
        sentential_form: Rule,
        nullable_set: Optional[set[Symbol]] = None,
    ) -> bool:
        """https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf"""
        if nullable_set is None:
            nullable_set = self.nullable()
        return all(sym in nullable_set for sym in sentential_form)

    def nullable(self, cache_key="nullable") -> set[Symbol]:
        """https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf"""

        if (cached := self.get_cached(cache_key)) is not None:
            return cast(set[Symbol], cached)

        nullable_set: set[Symbol] = {EMPTY}

        num_nullable = 0
        while True:
            for non_terminal, definition in self.items():
                should_be_added = any(
                    all(symbol in nullable_set for symbol in rule)
                    for rule in definition
                )
                already_present = non_terminal in nullable_set
                if should_be_added != already_present:
                    nullable_set.add(non_terminal)
            if len(nullable_set) == num_nullable:
                break
            num_nullable = len(nullable_set)
        self.cache(cache_key, nullable_set)
        return nullable_set

    def first_sentential_form(
        self,
        sentential_form: Sequence[Symbol],
        computing_first_set: Optional[dict[Symbol, set[Symbol]]] = None,
        cache_key="first_sf",
    ) -> set[Terminal]:
        if computing_first_set is None:
            computing_first_set = self.first()

        if not sentential_form:
            return {EMPTY}

        first_symbol, *rest = sentential_form

        if first_symbol not in computing_first_set:
            computing_first_set[first_symbol] = set()

        return (
            computing_first_set[first_symbol]
            | self.first_sentential_form(rest, computing_first_set, cache_key)
            if (first_symbol in self.nullable())
            else computing_first_set[first_symbol]
        )

    def first(self, cache_key="first") -> dict[Symbol, set[Terminal]]:
        if (cached := self.get_cached(cache_key)) is not None:
            return cast(FirstSet, cached)

        first_set = defaultdict(set)
        first_set.update({terminal: {terminal} for terminal in self.terminals})

        changed = True
        while changed:
            changed = False
            for non_terminal, sentential_forms in self.items():
                new_value = reduce(
                    set.union,
                    (
                        self.first_sentential_form(sentential_form, first_set)
                        for sentential_form in sentential_forms
                    ),
                    set(),
                )
                if new_value != first_set.get(non_terminal, None):
                    first_set[non_terminal] = cast(set[Terminal], new_value)
                    changed = True
        self.cache(cache_key, first_set)
        return first_set

    def follow(self, cache_key="follow"):
        if (follow_set := self.get_cached(cache_key)) is not None:
            return follow_set

        follow_set: FollowSet = defaultdict(set)
        follow_set[self._start_symbol] = {EOF}

        changed = True
        while changed:
            changed = False
            for A, definition in self.items():
                for rule in definition:
                    for index, B in rule.enumerate_variables():
                        first_in_suffix = self.first_sentential_form(rule[index + 1 :])
                        initial_size = len(follow_set[B])
                        follow_set[B] |= first_in_suffix - {EMPTY}
                        if EMPTY in first_in_suffix:
                            follow_set[B] |= follow_set[A]
                        if initial_size != len(follow_set[B]):
                            changed = True

        self.cache(cache_key, follow_set)
        return follow_set

    def build_ll1_parsing_table(self, cache_key="ll1_parsing_table") -> ParsingTable:
        if (cached := self.get_cached(cache_key)) is not None:
            return cached

        follow_set = self.follow()
        parsing_table = ParsingTable(self._terminals)

        for non_terminal, definition in self.items():
            for rule in definition:
                for terminal in self.first_sentential_form(rule):
                    if terminal is not EMPTY:
                        parsing_table[(non_terminal, terminal.id)] = rule
                if EMPTY in self.first_sentential_form(rule):
                    for terminal in follow_set[non_terminal]:
                        parsing_table[(non_terminal, terminal.id)] = rule

        self.cache(cache_key, parsing_table)
        return parsing_table

    def match(self, tokens: Sequence[Tokenizer.Token]):
        parsing_table = self.build_ll1_parsing_table()
        stack, token_index = [EOF, self._start_symbol], 0
        used_rules = []

        while stack:
            symbol = stack.pop()
            token = tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.id} but got {token}")
            else:
                symbol = cast(NonTerminal, symbol)
                if (rule := parsing_table.get((symbol, token.id))) is not None:
                    stack.extend(reversed(rule))
                    used_rules.append(ParseTableEntry(symbol, rule))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.id for terminal in self.first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert token_index >= len(tokens)
        return used_rules

    def __setitem__(self, key, value):
        raise Exception("Cannot modify grammar; use add_rule instead")
