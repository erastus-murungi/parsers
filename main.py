import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cache, reduce
from typing import Callable, Collection, Optional, Sequence, TypeGuard, cast

from rich import print as rprint
from rich.pretty import pretty_repr
from rich.traceback import install

from tokenizer import Tokenizer

install(show_locals=True)


class Symbol(ABC):
    def __init__(self, _id: str) -> None:
        self.id = _id

    def __hash__(self) -> int:
        return hash(self.id)

    @abstractmethod
    def __repr__(self):
        pass

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

    def __str__(self):
        return self.id


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
    __slots__ = ("augmented_start", "_start_symbol", "terminals")

    NON_TERMINAL_REGEX = r"<([A-Z][\w\']*)>"

    def __init__(self, start_symbol: Variable):
        super().__init__()
        self._start_symbol: Variable = start_symbol
        self.augmented_start = Variable("Grammar")
        self[self.augmented_start] = [SententialForm([self._start_symbol, EOF])]
        self.terminals: set[Terminal] = {EOF}

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

        self.terminals.update(
            (symbol for symbol in lhs if isinstance(symbol, Terminal))
        )
        if rhs not in self:
            self[rhs] = []
        if len(lhs) == 0:
            self[rhs].append(lhs.append_sentinel())
        else:
            self[rhs].append(lhs)

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

        start = SententialForm((self.augmented_start,))
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
        start = SententialForm((self.augmented_start,))
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

    def nullable(self) -> set[Symbol]:
        """https://fileadmin.cs.lth.se/cs/Education/EDAN65/2020/lectures/L05A.pdf"""
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

        return nullable_set

    def first(self) -> dict[Symbol, set[Terminal]]:
        first_set: dict[Symbol, set[Terminal]] = defaultdict(set)
        first_set.update({terminal: {terminal} for terminal in self.terminals})
        nullable_set = self.nullable()

        def first_sentential_form(sf: Sequence[Symbol]) -> set[Terminal]:
            if not sf:
                return set()
            s, *lam = sf
            return (
                first_set[s] | first_sentential_form(lam)
                if (s in nullable_set)
                else first_set[s]
            )

        changed = True
        while changed:
            changed = False
            for X, sentential_forms in self.items():
                new_value: set[Terminal] = reduce(
                    set.union, (map(first_sentential_form, sentential_forms)), set()
                )
                if new_value != first_set[X]:
                    first_set[X] = new_value
                    changed = True

        return first_set

    def follow(self):
        follow_set = {}
        for non_terminal in self.non_terminals:
            follow_set[non_terminal] = set()
        follow_set[self._start_symbol] = {EOF}

        first_set = self.first()
        nullable_set = self.nullable()

        @cache
        def first_sf(sf):
            s, *lam = sf
            return (
                first_set[s] | first_sf(lam)
                if (s in nullable_set and lam)
                else first_set[s]
            )

        @cache
        def nullable_sf(sf):
            return all(_sym in nullable_set for _sym in sf)

        done = False
        while not done:
            building = False
            for A in self.non_terminals:
                for B in self.non_terminals:
                    for prod in self[A]:
                        for index, sym in enumerate(prod):
                            if sym == B:
                                n = len(follow_set[B])
                                alpha, beta = prod[index:], prod[index + 1 :]
                                if beta:
                                    follow_set[B] |= first_sf(beta) - {EMPTY}
                                    if nullable_sf(beta):
                                        follow_set[B] |= follow_set[A]
                                    if len(follow_set[B]) > n:
                                        building = True
                                else:
                                    follow_set[B] |= follow_set[A]
                                    if len(follow_set[B]) > n:
                                        building = True
            done = not building
        return follow_set

    def parsing_table(self):
        first_set = self.first()
        nullable_set = self.nullable()
        follow_set = self.follow()

        @cache
        def first_sf(sf):
            if not sf:
                return set()
            s, *lam = sf
            return (
                first_set[s] | first_sf(SententialForm(lam))
                if (s in nullable_set and lam)
                else first_set[s]
            )

        @cache
        def nullable_sf(sf):
            return all(_sym in nullable_set for _sym in sf)

        parsing_table: dict[tuple, SententialForm] = {}
        for A, sentential_forms in self.items():
            for sentential_form in sentential_forms:
                for a in first_sf(sentential_form):
                    if a is not EMPTY:
                        if (A, a.id) in parsing_table:
                            raise ValueError(
                                f"grammar not LL(1); "
                                f"we have <{A}, {a.id}> mapping to {parsing_table[(A, a.id)]!s}, {sentential_form!s}"
                            )
                        parsing_table[(A, a.id)] = sentential_form
                if nullable_sf(
                    sentential_form
                ):  # if EMPTY in first_sf(sentential_form)
                    for b in follow_set[A]:
                        if (A, b.id) in parsing_table:
                            raise ValueError(
                                f"grammar not LL(1); "
                                f"we have <{A}, {b.id}> mapping to {parsing_table[(A, b.id)]!s}, {sentential_form!s}"
                            )
                        parsing_table[(A, b.id)] = sentential_form
        return dict(parsing_table)

    def match(self, tokens: Sequence[Tokenizer.Token]):
        parsing_table = self.parsing_table()
        stack, index = [EOF, self._start_symbol], 0
        rules = []

        while stack:
            symbol = stack.pop()
            token = tokens[index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
                    index += 1
                else:
                    raise SyntaxError(f"Expected {symbol.id} but got {token}")
            else:
                symbol = cast(Variable, Symbol)
                if (rule := parsing_table.get((symbol, token.id))) is not None:
                    stack.extend(reversed(rule))
                    rules.append(ProductionRule(symbol, rule))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.id for terminal in self.first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert index >= len(tokens)
        return rules

    @classmethod
    def from_string(cls, grammar, token_table) -> "ContextFreeGrammar":
        """
        Ad Hoc grammar parser
        """
        lines = grammar.strip().split("\n")
        start_symbol_str = re.match(cls.NON_TERMINAL_REGEX, lines[0])
        if start_symbol_str is None:
            raise ValueError("no start symbol found")
        start_symbol = Variable(start_symbol_str.group(1))
        cfg_obj = ContextFreeGrammar(start_symbol)

        for line in lines[1:]:
            lhs, rhs = re.split(r"::=", line)
            non_terminal_str = re.match(cls.NON_TERMINAL_REGEX, lhs.strip())
            if non_terminal_str is None:
                raise ValueError(
                    f"no non-terminal on rhs of {line}, check that syntax is correct"
                )
            non_terminal = Variable(non_terminal_str.group(1))

            parts = rhs.split("|")
            for part in parts:
                part = part.strip()
                matches = list(re.finditer(cls.NON_TERMINAL_REGEX, part))
                tokens_sequence: list[str] = []
                prev = 0
                for match in matches:
                    tokens_sequence.extend(part[prev : match.start()].split())
                    tokens_sequence.append(match.group(0))
                    prev = match.end()
                tokens_sequence.extend(part[prev:].split())

                sentential_form: list[Symbol] = []

                def bind_symbol(character: str):
                    return lambda tok: tok.lexeme == character

                for token in tokens_sequence:
                    if token == "<>":
                        break
                    elif token.startswith("<"):
                        # this is a non-terminal
                        sentential_form.append(Variable(token[1:-1]))
                    elif token.startswith("\\d"):
                        sentential_form.append(
                            Terminal(
                                "integer",
                                lambda tok: tok.token_type == "integer",
                            )
                        )
                    else:
                        sentential_form.append(
                            Terminal(token_table.get(token, token), bind_symbol(token))
                        )
                cfg_obj.add_production_rule(
                    non_terminal, SententialForm(sentential_form)
                )

        return cfg_obj


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
            <F> ::= (<E>) | \\d
    """

    cfg = ContextFreeGrammar.from_string(g, tk_table)
    rprint(pretty_repr(cfg))
    # rprint(pretty_repr(cfg.non_terminals))

    tks = Tokenizer("10 + (4*10) + 10", tk_table).get_tokens_no_whitespace()
    # rprint(pretty_repr(cfg.leftmost_top_down_parsing_dfs(tks)))
    rprint(pretty_repr(cfg.nullable()))
    rprint(pretty_repr(cfg.first()))
    rprint(pretty_repr(cfg.follow()))
    rprint(pretty_repr(cfg.parsing_table()))
    rprint(pretty_repr(cfg.match(tks)))
