from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeGuard, cast

from utils.tokenizer import Token


class Symbol(ABC):
    """A symbol in a grammar;
    Each is identified by a unique ID"""

    def __init__(self, _id: str) -> None:
        self.id = _id

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            raise NotImplemented
        return self.id == other.id


class Terminal(Symbol):
    def __init__(self, label: str, token_matcher: Callable[[Token], bool]):
        super().__init__(label)
        self._token_matcher = token_matcher

    def matches(self, token: Token) -> bool:
        return self._token_matcher(token)

    def __repr__(self):
        return f"[bold blue]{self.id}[/bold blue]"


class Marker(Terminal):
    def __repr__(self):
        return f"[bold cyan]{self.id}[/bold cyan]"


EOF = Marker("eof", lambda token: token.token_type == "eof")
EMPTY = Marker("ε", lambda token: True)


class NonTerminal(Symbol):
    def __repr__(self):
        return f"[bold red]<{self.id.capitalize()}>[/bold red]"


def all_terminals(symbols: Sequence[Symbol]) -> TypeGuard[Sequence[Terminal]]:
    return all(isinstance(symbol, Terminal) for symbol in symbols)


class Rule(list[Symbol]):
    def __init__(self, args: Optional[Iterable[Symbol]] = None):
        if args is None:
            args = []
        super().__init__(args)

    def __iter__(self):
        yield from filter(lambda token: token is not EMPTY, super().__iter__())

    def matches(self, tokens: Sequence[Token]) -> bool:
        if len(self) == len(tokens):
            if all_terminals(self):
                return all(
                    terminal.matches(token) for terminal, token in zip(self, tokens)
                )
        return False

    def perform_derivation(self, index, replacer: "Rule") -> "Rule":
        if not replacer:
            return Rule(self[:index] + self[index + 1 :])
        return Rule(self[:index] + replacer + self[index + 1 :])

    def append_marker(self, sentinel: Marker):
        return Rule(self + [sentinel])

    def enumerate_variables(self) -> Iterator[tuple[int, NonTerminal]]:
        for index, symbol in enumerate(self):
            if isinstance(symbol, NonTerminal):
                yield index, symbol

    def should_prune(
        self,
        tokens: Sequence[Token],
        seen: set["Rule"],
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

    def __hash__(self):
        return hash(tuple(self))

    def __len__(self):
        return super().__len__() - self.count(EMPTY)

    def __str__(self):
        return f'{"".join(str(item) for item in super().__iter__())}'

    def __repr__(self):
        return f'{"".join(repr(item) for item in super().__iter__())}'


@dataclass(frozen=True)
class ParseTableEntry:
    variable: NonTerminal
    sequence: Rule

    def __repr__(self):
        return f'{self.variable!r} => {"".join(f"{item!r}" for item in self.sequence.__iter__())}'

    def __iter__(self):
        yield from [self.variable, self.sequence]


@dataclass(slots=True, frozen=True)
class Node:
    form: Rule
    index: Optional[int] = None
    entry: Optional[ParseTableEntry] = None

    def update(
        self, index: int, entry: ParseTableEntry, replacement: Rule
    ) -> tuple["Node", "Node"]:
        return Node(self.form, index, entry), Node(replacement)


FollowSet = defaultdict[Symbol, set[Terminal]]
FirstSet = defaultdict[Symbol, set[Terminal]]
NullableSet = set[Symbol]


class Definition(list[Rule]):
    def __init__(self, rules: Optional[Iterable[Rule]] = None):
        if rules is None:
            rules = []
        super().__init__(rules)

    def __repr__(self):
        return " | ".join(repr(item) for item in self)
