from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Collection, Hashable, Optional, Sequence, TypeGuard, cast

from tokenizer import Token, Tokenizer, TokenType


class Symbol(Hashable):
    def __init__(self, label: str) -> None:
        self._label = label

    def __hash__(self) -> int:
        return hash(self._label)


class Terminal(Symbol):
    def __init__(self, token_type: TokenType):
        super().__init__(token_type.value)
        self.token_type = token_type

    def matches(self, token: Token) -> bool:
        return self.token_type == token.token_type

    def __repr__(self):
        return f"[bold blue]{self._label}[/bold blue]"


class Sentinel(Terminal):
    def __init__(self):
        super().__init__(TokenType.EOF)

    def matches(self, token: Token):
        return token.token_type == TokenType.EOF

    def __repr__(self):
        return f"[bold cyan]{self._label}[/bold cyan]"


class Variable(Symbol):
    def __repr__(self):
        return f"[bold red]<{self._label.capitalize()}>[/bold red]"


class SententialForm(tuple[Symbol]):
    def __init__(self, args):
        super().__new__(SententialForm, args)

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
        return SententialForm(self[:index] + replacer + self[index + 1 :])

    def enumerate_variables(self):
        for index, symbol in enumerate(self):
            if isinstance(symbol, Variable):
                yield index, symbol

    def should_prune(self, tokens: Sequence[Token], seen) -> bool:
        # if this is a sentential form we have explored, just ignore it
        if self in seen:
            return True

        # if the pattern is longer than the number of tokens then
        # we should prune
        if len(self) > len(tokens):
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


@dataclass(frozen=True)
class ProductionRule:
    variable: Variable
    sequence: SententialForm

    def __repr__(self):
        return (
            f'{repr(self.variable)} => {"".join(repr(item) for item in self.sequence)}'
        )

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


class ContextFreeGrammar(defaultdict[Variable, list[SententialForm]]):
    __slots__ = ("start_symbol", "terminals")

    def __init__(self, start_symbol: Variable):
        super().__init__(list)
        self.start_symbol: Variable = start_symbol
        self.terminals: set[Terminal] = set()

    def variables(self) -> Collection[Variable]:
        return self.keys()

    def add_production_rule(self, rhs: Variable, lhs: SententialForm) -> None:
        assert isinstance(lhs, SententialForm)
        self[rhs].append(lhs)
        self.terminals.update(
            (symbol for symbol in lhs if isinstance(symbol, Terminal))
        )

    def add_many_production_rule(
        self, rhs: Variable, sentential_forms: Sequence[SententialForm]
    ) -> None:
        for sentential_form in sentential_forms:
            self.add_production_rule(rhs, sentential_form)

    def __repr__(self) -> str:
        def rich_text(symbol: Symbol) -> str:
            return symbol.__repr__()

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

        start = SententialForm((self.start_symbol,))
        root = Node(start)
        sentential_forms: deque[ParseTreeSearchSpaceNode] = deque([((root,), start)])
        seen = set()

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
                    ).should_prune(tokens, seen):
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
        start = SententialForm((self.start_symbol,))
        root = Node(start)
        stack: list[ParseTreeSearchSpaceNode] = [((root,), start)]
        seen = set()

        n_iters = 0
        while stack and n_iters < max_iters:
            tree, sentential_form = stack.pop()

            seen.add(sentential_form)

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
                    ).should_prune(tokens, seen):
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


if __name__ == "__main__":
    from rich import print as rprint

    E, T, Op, integer, left_paren, right_paren = (
        Variable("E"),
        Variable("T"),
        Variable("Op"),
        Terminal(TokenType.INT),
        Terminal(TokenType.L_PAR),
        Terminal(TokenType.R_PAR),
    )
    plus, minus, mul, div = map(
        Terminal,
        [
            TokenType.ADD,
            TokenType.MULTIPLY,
            TokenType.SUBTRACT,
            TokenType.TRUE_DIV,
        ],
    )
    cfg = ContextFreeGrammar(E)
    cfg.add_production_rule(E, SententialForm([T]))
    cfg.add_production_rule(E, SententialForm((T, Op, E)))
    cfg.add_production_rule(T, SententialForm((integer,)))
    cfg.add_production_rule(T, SententialForm((left_paren, E, right_paren)))
    cfg.add_many_production_rule(
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
                Tokenizer("(10) * 10 + (10 / 3)").get_tokens_no_whitespace()
            )
        )
    )
