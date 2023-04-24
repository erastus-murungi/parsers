from abc import ABC
from collections import defaultdict
from typing import Iterable, Iterator, NamedTuple, Optional, Sequence, TypeGuard, cast


class Symbol(ABC):
    """A symbol in a grammar;
    Each is identified by a unique ID"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            raise NotImplemented
        return self.name == other.name


class Loc(NamedTuple):
    filename: str
    line: int
    col: int
    offset: int

    def __str__(self):
        return f"<{self.filename}:{self.line}:{self.col}>"


class Terminal(Symbol):
    """
    A token has three components:
    1) Its type
    2) A lexeme -- the substring of the source code it represents
    3) The location in code of the lexeme
    """

    def __init__(self, token_type: str, lexeme: str, loc: Loc):
        super().__init__(token_type)
        self.token_type = token_type
        self.lexeme = lexeme
        self.loc = loc

    @staticmethod
    def from_token_type(token_type: str, loc: Loc):
        """A convenient constructor to avoid the frequent pattern:
        Token(TokenType.X, TokenType.X.value, loc)"""
        return Terminal(token_type, token_type, loc)

    def matches(self, token: "Terminal") -> bool:
        if isinstance(token, Terminal):
            return self.token_type == token.token_type
        return False

    def __repr__(self):
        return f"[bold blue]{self.name}[/bold blue]"


class Marker(Terminal):
    def __repr__(self):
        return f"[bold cyan]{self.name}[/bold cyan]"

    def is_eof(self):
        return self.name == "eof"


class Empty(Marker):
    def matches(self, token: "Terminal") -> bool:
        return True


EMPTY = Empty("ε", "ε", Loc("(ε)", 0, 0, 0))
EOF = Marker("eof", "$", Loc("(eof)", 0, 0, 0))


class NonTerminal(Symbol):
    def __repr__(self):
        return f"[bold red]<{self.name}>[/bold red]"


def all_terminals(symbols: Sequence[Symbol]) -> TypeGuard[Sequence[Terminal]]:
    return all(isinstance(symbol, Terminal) for symbol in symbols)


class Expansion(tuple[Symbol]):
    def __new__(cls, args: Optional[Iterable[Symbol]] = None) -> "Expansion":
        if args is None:
            args = []
        return tuple.__new__(Expansion, args)  # type: ignore

    @staticmethod
    def empty() -> "Expansion":
        return Expansion({EMPTY})

    def __iter__(self):
        yield from filter(lambda token: token is not EMPTY, super().__iter__())

    def matches(self, tokens: Sequence[Terminal]) -> bool:
        if len(self) == len(tokens):
            if all_terminals(self):
                return all(
                    terminal.matches(token) for terminal, token in zip(self, tokens)
                )
        return False

    def perform_derivation(self, index, replacer: "Expansion") -> "Expansion":
        if not replacer:
            return Expansion(self[:index] + self[index + 1 :])
        return Expansion(self[:index] + replacer + self[index + 1 :])

    def enumerate_variables(self) -> Iterator[tuple[int, NonTerminal]]:
        for index, symbol in enumerate(self):
            if isinstance(symbol, NonTerminal):
                yield index, symbol

    def append(self, symbol: Symbol) -> "Expansion":
        return Expansion(self + (symbol,))

    def should_prune(
        self,
        tokens: Sequence[Terminal],
        seen: set["Expansion"],
        nullable_set: set[Symbol],
    ) -> bool:
        # if this is a sentential form we have explored, just ignore it
        if self in seen:
            return True

        # if we have more non-nullables than the number of tokens
        # we should prune
        if len(tuple(filter(lambda sym: sym not in nullable_set, self))) > len(tokens):
            return True

        # if we have a prefix of terminals which doesn't match the tokens
        # we should prune
        for symbol, token in zip(self, tokens):
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

    def __hash__(self):
        return hash(tuple(self))

    def __len__(self):
        return super().__len__() - self.count(EMPTY)

    def __str__(self):
        return f'{"".join(str(item) for item in super().__iter__())}'

    def __repr__(self):
        return f'{"".join(repr(item) for item in super().__iter__())}'


FollowSet = defaultdict[Symbol, set[Terminal]]
FirstSet = defaultdict[Symbol, set[Terminal]]
NullableSet = set[Symbol]
