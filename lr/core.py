from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Generic, Hashable, Iterable, Protocol, TypeVar, runtime_checkable

from prettytable import PrettyTable

from grammar import Grammar, NonTerminal, Symbol

FILENAME = "../graphs/state_graph"
DOT_FILEPATH = FILENAME + "." + "dot"
GRAPH_TYPE = "pdf"
OUTPUT_FILENAME = FILENAME + "." + GRAPH_TYPE


@runtime_checkable
class Completable(Protocol, Hashable):
    def completed(self) -> bool:
        ...


T = TypeVar("T", bound=Completable)


class LRState(list[T]):
    ids: dict["LRState", int] = defaultdict(count(1).__next__)

    def __init__(self, *items, cls: type[T]):
        assert issubclass(cls, Completable)
        self.type = cls
        assert all(
            isinstance(item, cls) for item in items
        ), "All items must be Completable"
        super().__init__()
        self.extend(items)

    @property
    def id(self):
        if not self:
            return 0
        return LRState.ids[self]

    def append(self, completable: T) -> None:
        if not isinstance(completable, self.type):
            raise TypeError(f"Expected {self.type}, got {type(completable)}")
        if completable not in self:
            super().append(completable)

    def extend(self, completables: Iterable[T]) -> None:
        for item in completables:
            self.append(item)

    def yield_finished(self):
        for item in self:
            if item.completed():
                yield item

    def yield_unfinished(self):
        for item in self:
            if not item.completed():
                yield item

    def copy(self) -> "LRState":
        return LRState(*self, cls=self.type)

    def __hash__(self):
        return hash(frozenset(self))

    def __eq__(self, other):
        return frozenset(self) == frozenset(other)

    def __str__(self):
        return "\n".join(str(item) for item in self)


class Action(ABC):
    """
    The super class of all possible actions that can be taken by a shift-reduce parser.
    """

    pass


@dataclass(frozen=True, slots=True)
class Reduce(Action):
    lhs: NonTerminal
    len_rhs: int

    def __str__(self):
        return f"Reduce({self.len_rhs!s})"


@dataclass(frozen=True, slots=True)
class Goto(Action, Generic[T]):
    state: LRState[T]

    def __str__(self):
        return f"Goto({self.state!s})"


@dataclass(frozen=True, slots=True)
class Accept(Action):
    pass


@dataclass(frozen=True, slots=True)
class Shift(Action, Generic[T]):
    """
    Advance input one token;
    push `state` on stack.

    Attributes
    ----------
    state: LRState[T]
        The LR state to push on the stack.

    """

    state: LRState[T]

    def __str__(self):
        return f"Shift(\n{self.state!s}\n)"


class LRTable(dict[tuple[LRState[T], str], Action], ABC):
    def __init__(self, grammar: Grammar, *, reduce: bool = True):
        super().__init__()
        self.grammar = grammar
        self.states: list[LRState[T]] = []
        self.reduce = reduce
        self.accept = None
        self.construct()

    def __hash__(self):
        return id(self)

    @abstractmethod
    def closure(self, state: LRState[T]):
        pass

    @abstractmethod
    def goto(self, state: LRState[T], sym: Symbol) -> LRState[T]:
        pass

    @abstractmethod
    def init_kernel(self) -> LRState[T]:
        pass

    @abstractmethod
    def compute_reduce_actions(self):
        pass

    @abstractmethod
    def construct(self):
        pass

    def to_pretty_table(self) -> str:
        syms: list[str] = [terminal.name for terminal in self.grammar.terminals] + [
            terminal.name for terminal in self.grammar.non_terminals
        ]

        pretty_table = PrettyTable()
        pretty_table.field_names = ["State"] + syms

        rows: list[list[str]] = []
        for state in self.states:
            row: list[str] = [str(state.id)]
            for sym in syms:
                match self.get((state, sym), None):
                    case Goto(state):
                        row.append(f"goto {state.id}")
                    case Shift(state):
                        row.append(f"shift {state.id}")
                    case Reduce(name, len_rule):
                        row.append(f"reduce {name!s} {{{len_rule!s}}})")
                    case Accept():
                        row.append("accept")
                    case _:
                        row.append("")
            rows.append(row)

        rows.sort(key=lambda r: r[0])
        pretty_table.add_rows(rows)

        return pretty_table.get_string()
