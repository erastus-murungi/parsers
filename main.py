import re
from abc import ABC
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import reduce
from itertools import product
from typing import Callable, Collection, Iterator, Optional, Sequence, TypeGuard, cast

from rich import print as rprint
from rich.pretty import pretty_repr
from rich.traceback import install

from tokenizer import Tokenizer

install(show_locals=True)


class Symbol(ABC):
    """A symbol in a grammar; Each is identified by a unique ID"""

    def __init__(self, _id: str) -> None:
        self.id = _id

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self):
        return self.id


class Terminal(Symbol):
    def __init__(
        self, label: str, matching_function: Callable[[Tokenizer.Token], bool]
    ):
        super().__init__(label)
        self.matching_function = matching_function

    def matches(self, token: Tokenizer.Token) -> bool:
        return self.matching_function(token)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f"[bold blue]{self.id}[/bold blue]"


class Marker(Terminal):
    def __repr__(self):
        return f"[bold cyan]{self.id}[/bold cyan]"

    def __str__(self):
        return self.id


EOF = Marker("eof", lambda token: token.token_type == "eof")
EMPTY = Marker("ε", lambda token: True)


class Variable(Symbol):
    def __repr__(self):
        return f"[bold red]<{self.id.capitalize()}>[/bold red]"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return self.id


class SententialForm(tuple[Symbol]):
    def __init__(self, args):
        super().__new__(SententialForm, args)

    def __iter__(self):
        yield from filter(lambda token: token is not EMPTY, super().__iter__())

    def matches(self, tokens: Sequence[Tokenizer.Token]) -> bool:
        def all_terminals(symbols: Sequence[Symbol]) -> TypeGuard[Sequence[Terminal]]:
            return all(isinstance(symbol, Terminal) for symbol in symbols)

        if len(self) == len(tokens):
            if all_terminals(self):
                return all(
                    terminal.matches(token) for terminal, token in zip(self, tokens)
                )
        return False

    def perform_derivation(self, index, replacer: "SententialForm") -> "SententialForm":
        if not replacer:
            return SententialForm(self[:index] + self[index + 1 :])
        return SententialForm(self[:index] + replacer + self[index + 1 :])

    def append_sentinel(self):
        return SententialForm(self + (EMPTY,))

    def enumerate_variables(self):
        for index, symbol in enumerate(self):
            if isinstance(symbol, Variable):
                yield index, symbol

    def should_prune(
        self,
        tokens: Sequence[Tokenizer.Token],
        seen: set["SententialForm"],
        nullable_set,
    ) -> bool:
        # if this is a sentential form we have explored, just ignore it
        if self in seen:
            return True

        if len(tuple(filter(lambda sym: sym not in nullable_set, self))) > len(tokens):
            return True

        # if we have a prefix of terminals which doesn't match the tokens
        # we should prune
        for (symbol, token) in zip(self, tokens):
            if isinstance(symbol, Terminal):
                if not symbol.matches(token):
                    return True
            else:
                break
        else:
            # if the sentential form is a PROPER prefix of the tokens
            # we should prune
            return len(self) != len(tokens)

        # if any of the tokens in the sentential form is not in the tokens,
        # we should prune
        for terminal in filter(lambda item: isinstance(item, Terminal), self):
            if not any(cast(Terminal, terminal).matches(token) for token in tokens):
                return True
        return False

    def __len__(self):
        return super().__len__() - self.count(EMPTY)

    def __str__(self):
        return f'`{"".join(str(item) for item in self)}`'

    def __repr__(self):
        return f'`{"".join(repr(item) for item in self)}`'


@dataclass(frozen=True)
class ProductionRule:
    variable: Variable
    sequence: SententialForm

    def __repr__(self):
        return f'{self.variable!r} => {"".join(f"{item!r}" for item in self.sequence)}'

    def __iter__(self):
        yield from [self.variable, self.sequence]


@dataclass(slots=True, frozen=True)
class Node:
    form: SententialForm
    index: Optional[int] = None
    production_rule: Optional[ProductionRule] = None

    def update(
        self, index: int, production_rule, replacement: SententialForm
    ) -> tuple["Node", "Node"]:
        return Node(self.form, index, production_rule), Node(replacement)


ParseTreeSearchSpaceNode = tuple[tuple[Node, ...], SententialForm]


class ContextFreeGrammar(dict[Variable, list[SententialForm]]):
    __slots__ = (
        "_augmented_start",
        "_start_symbol",
        "_terminals",
        "_current_time",
        "_caches",
    )

    NON_TERMINAL_REGEX = r"<([A-Z][\w\']*)>"

    def __init__(self, start_symbol: Variable):
        super().__init__()
        self._start_symbol: Variable = start_symbol
        self._augmented_start = Variable("Grammar")
        self[self._augmented_start] = [SententialForm([self._start_symbol, EOF])]
        self._terminals: set[Terminal] = {EOF}
        self._current_time = 0
        self._caches = {}

    def __len__(self):
        return super().__len__() - 1

    @property
    def non_terminals(self) -> Collection[Variable]:
        return set(self.keys())

    def add_production_rule(self, rhs: Variable, lhs: SententialForm) -> None:
        assert isinstance(rhs, Variable)
        if EOF in lhs:
            raise ValueError(
                "you are not allowed to explicit add an EOF token, "
                "it is implicitly added by the grammar object"
            )
        if EMPTY in lhs:
            raise ValueError(
                "you are not allowed to explicit add a sentinel, "
                "pass in empty SententialForm instead e.g "
                "`add_production_rule(var, SententialForm())`"
            )

        self._terminals.update(
            (symbol for symbol in lhs if isinstance(symbol, Terminal))
        )
        if rhs not in self:
            self[rhs] = []
        if len(lhs) == 0:
            self[rhs].append(lhs.append_sentinel())
        else:
            self[rhs].append(lhs)
        self._current_time += 1

    def add_many_production_rules(
        self, rhs: Variable, sentential_forms: Sequence[SententialForm]
    ) -> None:
        for sentential_form in sentential_forms:
            self.add_production_rule(rhs, sentential_form)

    def __repr__(self) -> str:
        def rich_text(symbol: Symbol) -> str:
            return repr(symbol)

        def line(rhs: Variable, lhs: list[SententialForm]) -> str:
            return f'{rich_text(rhs)} => {" | ".join("".join(map(rich_text, item)) for item in lhs)}'

        return "\n".join(line(rhs, lhs) for rhs, lhs in self.items())

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

        start = SententialForm((self._augmented_start,))
        root = Node(start)
        sentential_forms: deque[ParseTreeSearchSpaceNode] = deque([((root,), start)])
        seen = set()
        nullable_set = self.nullable()

        n_iters = 0
        while sentential_forms and n_iters < max_iters:
            tree, sentential_form = sentential_forms.popleft()

            seen.add(sentential_form)

            rprint(repr(sentential_form))

            if sentential_form.matches(tokens):
                return tree

            for index, symbol in sentential_form.enumerate_variables():
                for replacement in self[symbol]:
                    if (
                        next_form := sentential_form.perform_derivation(
                            index, replacement
                        )
                    ).should_prune(tokens, seen, nullable_set):
                        continue
                    sentential_forms.append(
                        (
                            tree[:-1]
                            + tree[-1].update(
                                index, ProductionRule(symbol, replacement), next_form
                            ),
                            next_form,
                        )
                    )

            n_iters += 1

    def leftmost_top_down_parsing_dfs(
        self, tokens: list[Tokenizer.Token], max_iters: int = 1000_000
    ):
        start = SententialForm((self._augmented_start,))
        root = Node(start)
        stack: list[ParseTreeSearchSpaceNode] = [((root,), start)]
        seen = set()
        nullable_set = self.nullable()

        n_iters = 0
        while stack and n_iters < max_iters:
            tree, sentential_form = stack.pop()

            rprint(repr(sentential_form))

            seen.add(sentential_form)

            if sentential_form.matches(tokens):
                return tree

            next_in_stack = []

            for index, symbol in sentential_form.enumerate_variables():
                for replacement in self[symbol]:
                    if (
                        next_form := sentential_form.perform_derivation(
                            index, replacement
                        )
                    ).should_prune(tokens, seen, nullable_set):
                        continue

                    next_in_stack.append(
                        (
                            tree[:-1]
                            + tree[-1].update(
                                index, ProductionRule(symbol, replacement), next_form
                            ),
                            next_form,
                        )
                    )
            stack.extend(reversed(next_in_stack))
            n_iters += 1

    def get_cached(self, function_name: str):
        if function_name not in self._caches:
            return None
        compute_time, cached = self._caches[function_name]
        if compute_time == self._current_time:
            return cached
        return None

    def cache(self, function_name: str, data):
        self._caches[function_name] = (self._current_time, data)

    def nullable_sentential_form(
        self,
        sentential_form: SententialForm,
        nullable_set: Optional[set[Symbol]] = None,
    ) -> bool:
        """https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf"""
        if nullable_set is None:
            nullable_set = self.nullable()
        return all(sym in nullable_set for sym in sentential_form)

    def nullable(self, cache_key="nullable") -> set[Symbol]:
        """https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf"""

        if (nullable_set := self.get_cached(cache_key)) is not None:
            return nullable_set

        nullable_set: set[Symbol] = {EMPTY}

        num_nullable = 0
        while True:
            for X, sentential_forms in self.items():
                should_be_added = any(
                    all(sym in nullable_set for sym in sentential_form)
                    for sentential_form in sentential_forms
                )
                already_present = X in nullable_set
                if should_be_added != already_present:
                    nullable_set.add(X)
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
            return set()
        first_symbol, *rest = sentential_form
        return (
            computing_first_set[first_symbol]
            | self.first_sentential_form(rest, computing_first_set, cache_key)
            if (first_symbol in self.nullable())
            else computing_first_set[first_symbol]
        )

    def first(self, cache_key="first") -> dict[Symbol, set[Terminal]]:
        if (first_set := self.get_cached(cache_key)) is not None:
            return first_set

        first_set: dict[Symbol, set[Terminal]] = defaultdict(set)
        first_set.update({terminal: {terminal} for terminal in self._terminals})

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
                if new_value != first_set[non_terminal]:
                    first_set[non_terminal] = new_value
                    changed = True
        self.cache(cache_key, first_set)
        return first_set

    def follow(self, cache_key="follow"):
        if (follow_set := self.get_cached(cache_key)) is not None:
            return follow_set

        follow_set = {}
        for non_terminal in self.non_terminals:
            follow_set[non_terminal] = set()
        follow_set[self._start_symbol] = {EOF}

        building = True
        while building:
            building = False
            for non_terminal_a, non_terminal_b in product(self.non_terminals, repeat=2):
                for sentential_form in self[non_terminal_a]:
                    for index, symbol in enumerate(sentential_form):
                        if symbol == non_terminal_b:
                            initial_follow_set_size = len(follow_set[non_terminal_b])
                            if suffix_sentential := SententialForm(
                                sentential_form[index + 1 :]
                            ):
                                follow_set[
                                    non_terminal_b
                                ] |= self.first_sentential_form(suffix_sentential) - {
                                    EMPTY
                                }
                                if self.nullable_sentential_form(suffix_sentential):
                                    follow_set[non_terminal_b] |= follow_set[
                                        non_terminal_a
                                    ]
                                if (
                                    len(follow_set[non_terminal_b])
                                    > initial_follow_set_size
                                ):
                                    building = True
                            else:
                                follow_set[non_terminal_b] |= follow_set[non_terminal_a]
                                if (
                                    len(follow_set[non_terminal_b])
                                    > initial_follow_set_size
                                ):
                                    building = True
        self.cache(cache_key, follow_set)
        return follow_set

    def build_ll1_parsing_table(self, cache_key="ll1_parsing_table"):
        if (parsing_table := self.get_cached(cache_key)) is not None:
            return parsing_table

        follow_set = self.follow()

        parsing_table: dict[tuple, SententialForm] = {}
        for non_terminal, sentential_forms in self.items():
            for sentential_form in sentential_forms:
                for terminal in self.first_sentential_form(sentential_form):
                    if terminal is not EMPTY:
                        if (non_terminal, terminal.id) in parsing_table:
                            raise ValueError(
                                f"grammar not LL(1); "
                                f"we have <{non_terminal}, {terminal.id}> "
                                f"mapping to {parsing_table[(non_terminal, terminal.id)]!s}, {sentential_form!s}"
                            )
                        parsing_table[(non_terminal, terminal.id)] = sentential_form
                if self.nullable_sentential_form(
                    sentential_form
                ):  # if EMPTY in first_sf(sentential_form)
                    for terminal in follow_set[non_terminal]:
                        if (non_terminal, terminal.id) in parsing_table:
                            raise ValueError(
                                f"grammar not LL(1); "
                                f"we have <{non_terminal}, {terminal.id}> "
                                f"mapping to {parsing_table[(non_terminal, terminal.id)]!s}, {sentential_form!s}"
                            )
                        parsing_table[(non_terminal, terminal.id)] = sentential_form

        parsing_table = dict(parsing_table)
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
                    token_index += 1
                else:
                    raise SyntaxError(f"Expected {symbol.id} but got {token}")
            else:
                symbol = cast(Variable, symbol)
                if (rule := parsing_table.get((symbol, token.id))) is not None:
                    stack.extend(reversed(rule))
                    used_rules.append(ProductionRule(symbol, rule))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.id for terminal in self.first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert token_index >= len(tokens)
        return used_rules

    @classmethod
    def from_string(
        cls, grammar_str: str, defined_tokens: dict[str, str]
    ) -> "ContextFreeGrammar":
        """
        Ad Hoc grammar parser
        """

        def iter_symbol_tokens(input_str: str) -> Iterator[str]:
            input_str = input_str.strip()
            non_terminal_matches = list(re.finditer(cls.NON_TERMINAL_REGEX, input_str))
            start_index = 0
            for non_terminal_match in non_terminal_matches:
                yield from input_str[start_index : non_terminal_match.start()].split()
                yield non_terminal_match.group(0)
                start_index = non_terminal_match.end()
            yield input_str[start_index:].split()

        lines = grammar_str.strip().split("\n")
        if (start_symbol_match := re.match(cls.NON_TERMINAL_REGEX, lines[0])) is None:
            raise ValueError("no start symbol found")
        start_symbol = Variable(start_symbol_match.group(1))
        cfg = ContextFreeGrammar(start_symbol)

        for line in lines[1:]:
            lhs, rhs = re.split(r"::=", line)
            if (
                lhs_non_terminal_match := re.match(cls.NON_TERMINAL_REGEX, lhs.strip())
            ) is None:
                raise ValueError(
                    f"no non-terminal on rhs of {line}, check that syntax is correct"
                )
            lhs_non_terminal = Variable(lhs_non_terminal_match.group(1))

            for rhs_sentential_token in rhs.split("|"):
                sentential_form_symbols: list[Symbol] = []

                def bind_lexeme(lexeme: str):
                    return lambda token: token.lexeme == lexeme

                def bind_token_type(token_type: str):
                    return lambda token: token.token_type == token_type

                for lexeme in iter_symbol_tokens(rhs_sentential_token):
                    if lexeme == "<>":
                        break
                    elif lexeme.startswith("<"):
                        # this is a non-terminal
                        sentential_form_symbols.append(Variable(lexeme[1:-1]))
                    else:
                        # keywords
                        if lexeme in ("integer", "float", "whitespace"):
                            sentential_form_symbols.append(
                                Terminal(lexeme, bind_token_type(lexeme))
                            )
                        else:
                            sentential_form_symbols.append(
                                Terminal(
                                    defined_tokens.get(lexeme, lexeme),
                                    bind_lexeme(lexeme),
                                )
                            )
                cfg.add_production_rule(
                    lhs_non_terminal, SententialForm(sentential_form_symbols)
                )

        return cfg


if __name__ == "__main__":

    # g = """
    #         <E>
    #         <E> ::= \\d
    #         <E> ::= (<E> <Op> <E>)
    #         <Op> ::= +
    #         <Op> ::= *
    # """
    #
    # cfg = ContextFreeGrammar.from_string(g)
    # rprint(pretty_repr(cfg))
    # rprint(pretty_repr(cfg.non_terminals))
    #
    # tokens = Tokenizer("((10 + 7) * 7)").get_tokens_no_whitespace()
    # rprint(pretty_repr(cfg.leftmost_top_down_parsing_dfs(tokens)))
    # rprint(pretty_repr(cfg.nullable()))
    # rprint(pretty_repr(cfg.first()))
    # rprint(pretty_repr(cfg.follow()))
    # rprint(pretty_repr(cfg.parsing_table()))
    # rprint(pretty_repr(cfg.match(tokens)))

    # g = """
    #         <A>
    #         <A> ::= <A>b
    #         <A> ::= c
    # """
    # g = """
    #         <A>
    #         <A> ::= c<B>
    #         <B> ::= <>
    #         <B> ::= b<B>
    # """
    tk_table = {"+": "+", "(": "(", "*": "*", ")": ")"}
    g = """ 
            <E>
            <E> ::= <T><E'>
            <E'> ::= + <T><E'>|<>
            <T> ::= <F><T'>
            <T'> ::= * <F><T'> | <>
            <F> ::= (<E>) | integer | float
    """

    cfg = ContextFreeGrammar.from_string(g, tk_table)
    rprint(pretty_repr(cfg))
    # rprint(pretty_repr(cfg.non_terminals))

    tks = Tokenizer("(10.44 + 19444 * 0xF)", tk_table).get_tokens_no_whitespace()
    rprint(pretty_repr(cfg.leftmost_top_down_parsing_dfs(tks)))
    rprint(pretty_repr(cfg.nullable()))
    rprint(pretty_repr(cfg.first()))
    rprint(pretty_repr(cfg.follow()))
    rprint(pretty_repr(cfg.build_ll1_parsing_table()))
    rprint(pretty_repr(cfg.match(tks)))
