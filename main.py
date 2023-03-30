from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Collection, Hashable, Self, Sequence

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
        self.only_terminals = all(isinstance(symbol, Terminal) for symbol in self)

    def matches(self, tokens: Sequence[Token]) -> bool:
        if not self.only_terminals:
            return False
        if len(self) != len(tokens):
            return False
        return all(terminal.matches(token) for terminal, token in zip(self, tokens))

    def perform_derivation(self, index, replacer: Self) -> Self:
        return SententialForm(self[:index] + replacer + self[index + 1 :])

    def enumerate_variables(self):
        for index, symbol in enumerate(self):
            if isinstance(symbol, Variable):
                yield index, symbol

    def should_prune(self, tokens: Sequence[Token]) -> bool:
        if len(self) > len(tokens):
            return True
        for terminal in filter(lambda item: isinstance(item, Terminal), self):
            if not any(terminal.matches(token) for token in tokens):
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


@dataclass(slots=True)
class Node:
    ...


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

    def naive_top_down_parsing(self, tokens: list[Token], max_iters: int = 1000_000):
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

        :param tokens:
        :param max_iters:
        :return:
        """
        sentential_forms: deque[tuple[tuple, SententialForm]] = deque(
            [((), SententialForm((self.start_symbol,)))]
        )

        n_iters = 0
        while sentential_forms and n_iters < max_iters:
            tree, sentential_form = sentential_forms.popleft()

            if sentential_form.matches(tokens):
                return tree

            for index, symbol in sentential_form.enumerate_variables():
                for replacement in self[symbol]:
                    sentential_forms.append(
                        (
                            tree + (sentential_form,),
                            sentential_form.perform_derivation(index, replacement),
                        )
                    )

            n_iters += 1

    def leftmost_top_down_parsing(self, tokens: list[Token], max_iters: int = 1000_000):
        stack: list[tuple[tuple, SententialForm]] = [
            ((), SententialForm((self.start_symbol,)))
        ]

        n_iters = 0

        while stack and n_iters < max_iters:
            tree, sentential_form = stack.pop()
            rprint(repr(sentential_form))

            if sentential_form.matches(tokens):
                return tree

            next_in_stack = []

            for index, symbol in sentential_form.enumerate_variables():
                for replacement in self[symbol]:
                    if (
                        next_form := sentential_form.perform_derivation(
                            index, replacement
                        )
                    ).should_prune(tokens):
                        continue

                    next_in_stack.append(
                        (
                            tree + (sentential_form,),
                            next_form,
                        )
                    )
            stack.extend(reversed(next_in_stack))
            n_iters += 1


if __name__ == "__main__":
    from rich import print as rprint

    E, T, plus, integer, left_paren, right_paren = (
        Variable("E"),
        Variable("T"),
        Terminal(TokenType.ADD),
        Terminal(TokenType.INT),
        Terminal(TokenType.L_PAR),
        Terminal(TokenType.R_PAR),
    )
    cfg = ContextFreeGrammar(E)
    cfg.add_production_rule(E, SententialForm([T]))
    cfg.add_production_rule(E, SententialForm((T, plus, E)))
    cfg.add_production_rule(T, SententialForm((integer,)))
    cfg.add_production_rule(T, SententialForm((left_paren, E, right_paren)))

    rprint(repr(cfg))
    rprint(
        repr(
            cfg.naive_top_down_parsing(
                Tokenizer("10 + 10 + (12)").get_tokens_no_whitespace()
            )
        )
    )
