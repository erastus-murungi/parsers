from abc import ABC
from dataclasses import dataclass
from typing import Hashable, Iterable, Protocol, TypeVar, runtime_checkable

from core import NonTerminal, Rule


@runtime_checkable
class Completable(Protocol, Hashable):
    def completed(self) -> bool:
        ...


T = TypeVar("T", bound=Completable)


class State(list[T]):
    def __init__(self, *items, cls: type[T]):
        self.type = cls
        assert all(
            isinstance(item, cls) for item in items
        ), "All items must be Completable"
        super().__init__()
        self.extend(items)

    def append(self, completable: T) -> None:
        if not isinstance(completable, Completable):
            raise TypeError(f"Expected EarleyItem, got {type(completable)}")
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

    def copy(self) -> "State":
        return State(*self, cls=self.type)

    def __hash__(self):
        return hash(tuple(self))

    def __str__(self):
        return "\n".join(str(item) for item in self)


class Action(ABC):
    pass


@dataclass(frozen=True, slots=True)
class Reduce(Action):
    lhs: NonTerminal
    rule: Rule

    def __str(self):
        return f"Reduce({self.lhs!s} -> {' '.join(str(sym) for sym in self.rule)})"


@dataclass(frozen=True, slots=True)
class Goto(Action):
    state: State[T]

    def __str__(self):
        return f"Goto({self.state!s})"


@dataclass(frozen=True, slots=True)
class Accept(Action):
    pass


@dataclass(frozen=True, slots=True)
class Shift(Action):
    state: State[T]

    def __str__(self):
        return f"Shift(\n{self.state!s}\n)"
