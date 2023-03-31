from abc import abstractmethod, ABC
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import reduce, cache
from typing import Collection, Hashable, Optional, Sequence, TypeGuard, cast

from tokenizer import Token, Tokenizer, TokenType


class Symbol(Hashable, ABC):
    def __init__(self, label: str) -> None:
        self._label = label

    def __hash__(self) -> int:
        return hash(self._label)

    @abstractmethod
    def __rich_repr__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class Terminal(Symbol):
    def __init__(self, token_type: TokenType):
        super().__init__(token_type.value)
        self.token_type = token_type

    def matches(self, token: Token) -> bool:
        return self.token_type == token.token_type

    def __rich_repr__(self):
        yield self._label

    def __repr__(self):
        return f"[bold blue]{self._label}[/bold blue]"


class _Empty(Terminal):
    def __init__(self):
        super().__init__(TokenType.EMPTY)

    def matches(self, token: Token):
        return True

    def __repr__(self):
        return f"[bold cyan]ε[/bold cyan]"


class _EndOfFile(Terminal):
    def __init__(self):
        super().__init__(TokenType.EOF)

    def matches(self, token: Token):
        return token.token_type == TokenType.EOF

    def __repr__(self):
        return f"[bold cyan]$[/bold cyan]"


EOF = _EndOfFile()
EMPTY = _Empty()


class Variable(Symbol):
    def __rich_repr__(self):
        yield f"<{self._label.capitalize()}>"

    def __repr__(self):
        return f"[bold red]<{self._label.capitalize()}>[/bold red]"


class SententialForm(tuple[Symbol]):
    def __init__(self, args):
        super().__new__(SententialForm, args)

    def __iter__(self):
        yield from filter(lambda token: token is not EMPTY, super().__iter__())

    def matches(self, tokens: Sequence[Token]) -> bool:
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
        self, tokens: Sequence[Token], seen: set["SententialForm"], nullable_set
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
        return self.keys()

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

    def __rich_repr__(self) -> str:
        def rich_text(symbol: Symbol) -> str:
            return tuple(symbol.__rich_repr__())[0]

        def line(rhs: Variable, lhs: list[SententialForm]) -> str:
            return f'{rich_text(rhs)} => {" | ".join("".join(map(rich_text, item)) for item in lhs)}\n'

        for rhs, lhs in self.items():
            yield line(rhs, lhs)

    def __repr__(self) -> str:
        def rich_text(symbol: Symbol) -> str:
            return repr(symbol)

        def line(rhs: Variable, lhs: list[SententialForm]) -> str:
            return f'{rich_text(rhs)} => {" | ".join("".join(map(rich_text, item)) for item in lhs)}'

        return "\n".join(line(rhs, lhs) for rhs, lhs in self.items())

    def leftmost_top_down_parsing_bfs(
        self, tokens: list[Token], max_iters: int = 1000_000
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
        self, tokens: list[Token], max_iters: int = 1000_000
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
        nullable_set = {EMPTY}

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
        first_set = defaultdict(set)
        first_set.update({terminal: {terminal} for terminal in self.terminals})
        nullable_set = self.nullable()

        def first_sentential_form(sf):
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
                new_value = reduce(
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

        M: dict[tuple, set] = defaultdict(set)
        for A, sentential_forms in self.items():
            for sentential_form in sentential_forms:
                FIRST = first_sf(sentential_form)
                for a in FIRST:
                    if a is not EMPTY:
                        M[(A, a)].add(ProductionRule(A, sentential_form))
                if EMPTY in FIRST:
                    for b in follow_set[A]:
                        M[(A, b)].add(ProductionRule(A, sentential_form))

        for A in self.non_terminals:
            for a in self.terminals:
                if not M[(A, a)]:
                    M[(A, a)] = None
        return M


if __name__ == "__main__":
    from rich import print as rprint
    from rich.pretty import pretty_repr

    E, T, Op, Sign, integer, left_paren, right_paren = (
        Variable("E"),
        Variable("T"),
        Variable("Op"),
        Variable("Sign"),
        Terminal(TokenType.INT),
        Terminal(TokenType.L_PAR),
        Terminal(TokenType.R_PAR),
    )
    plus, minus, mul, div = map(
        Terminal,
        [
            TokenType.ADD,
            TokenType.SUBTRACT,
            TokenType.MULTIPLY,
            TokenType.TRUE_DIV,
        ],
    )

    cfg = ContextFreeGrammar(E)
    cfg.add_production_rule(E, SententialForm([T]))
    cfg.add_production_rule(E, SententialForm((T, Op, E)))
    cfg.add_production_rule(
        T,
        SententialForm(
            (
                Sign,
                integer,
            )
        ),
    )
    cfg.add_production_rule(T, SententialForm((left_paren, E, right_paren)))
    cfg.add_production_rule(Sign, SententialForm([minus]))
    cfg.add_production_rule(Sign, SententialForm([plus]))
    cfg.add_production_rule(Sign, SententialForm([]))
    cfg.add_many_production_rules(
        Op,
        [
            SententialForm((plus,)),
            SententialForm((minus,)),
            SententialForm((mul,)),
            SententialForm((div,)),
        ],
    )

    rprint(repr(cfg))
    rprint(
        repr(
            cfg.leftmost_top_down_parsing_dfs(
                Tokenizer("+10 + 7 + 1").get_tokens_no_whitespace()
            )
        )
    )
    rprint(pretty_repr(cfg.nullable()))
    rprint(pretty_repr(cfg.first()))
    rprint(pretty_repr(cfg.follow()))
    rprint(pretty_repr(cfg.parsing_table()))
