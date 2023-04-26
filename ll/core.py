from itertools import islice
from typing import Iterable, Iterator, MutableSet, Self

from more_itertools import partition

from grammar import EMPTY, EOF, Terminal


def k_length(terminals: Iterable[Terminal], k: int) -> int:
    # Returns the k-length, i.e. the number of symbols that contributes to lookahead sizes
    k_len = 0
    for terminal in terminals:
        if k_len >= k:
            break
        k_len += 1
        if terminal is EOF:
            break
    return k_len


class TerminalString(tuple[Terminal]):
    k: int

    def __new__(cls, terminals: Iterable[Terminal], k: int):
        self = tuple.__new__(TerminalString, islice(terminals, k))  # type: ignore
        self.k = k
        return self

    def is_complete(self, k: int):
        return not self.is_eps() and len(self) >= k or (self and self[-1] is EOF)

    @staticmethod
    def eps():
        return TerminalString([EMPTY], 1)

    @staticmethod
    def eof():
        return TerminalString([EOF], 1)

    def is_eps(self):
        return len(self) == 1 and self[0] is EMPTY

    # Concatenates two collections with respect to the rules of k-concatenation
    def k_concat(self, other: "TerminalString", k: int) -> "TerminalString":
        if other.is_eps():
            # w + ε = w
            return self

        terminals = list(self)
        if self.is_eps():
            # ε + w = w
            # Remove possible epsilon terminal
            terminals.clear()

        if self.is_complete(k):
            # k: w would be the same as k: (w + x)
            return TerminalString(terminals, k)

        my_k_len = k_length(terminals, k)
        to_take = k_length(other, k - my_k_len)
        terminals.extend(other[:to_take])
        return TerminalString(terminals, k)

    def __repr__(self):
        return "".join(repr(item) for item in self)


class TerminalStrings(MutableSet[TerminalString]):
    k: int

    def __init__(self, items: Iterable[TerminalString], k: int):
        self._items: set[TerminalString] = set()
        self.k = k
        for item in items:
            self.add(item)

    @staticmethod
    def intersection(*args: tuple["TerminalStrings", ...]):
        assert args
        assert all(isinstance(ts, TerminalStrings) for ts in args)
        first = args[0]
        assert all(ts.k == first.k for ts in args)
        return TerminalStrings(set.intersection(*(ts._items for ts in args)), first.k)

    @staticmethod
    def of(terminal_string: TerminalString, k: int):
        return TerminalStrings([terminal_string], k)

    @staticmethod
    def eps(k):
        assert k >= 1
        return TerminalStrings([TerminalString.eps()], k)

    @staticmethod
    def eof(k):
        assert k >= 1
        return TerminalStrings([TerminalString.eof()], k)

    @staticmethod
    def empty(k):
        assert k >= 0
        return TerminalStrings([], k)

    def increment_k(self, k: int) -> "TerminalStrings":
        assert k >= self.k
        return TerminalStrings(self._items, k)

    def is_complete(self):
        return all(item.is_complete(self.k) for item in self)

    def k_concat(self, value: "TerminalStrings", k) -> Self:
        if not self.is_complete():
            incomplete, complete = partition(lambda x: x.is_complete(k), self)
            self._items = set(complete)
            for terminal_string in incomplete:
                for other in value:
                    self.add(terminal_string.k_concat(other, k))
        return self

    def add(self, value: TerminalString) -> None:
        if not isinstance(value, TerminalString):
            raise TypeError(f"Expected TerminalString, got {value}")
        assert value.k <= self.k
        self._items.add(value)

    def union(self, values: "TerminalStrings") -> "TerminalStrings":
        if not isinstance(values, TerminalStrings):
            raise TypeError(f"Expected TerminalStrings, got {values}")
        assert (value.k <= self.k for value in values)
        return TerminalStrings(self._items | values._items, self.k)

    def __or__(self, other):
        if not isinstance(other, TerminalStrings):
            raise TypeError(f"Expected TerminalStrings, got {other}")
        assert other.k <= self.k
        return self.union(other)

    def discard(self, value: TerminalString) -> None:
        raise NotImplementedError

    def __contains__(self, x: object) -> bool:
        return x in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[TerminalString]:
        return iter(self._items)

    def __repr__(self):
        return "{" + ", ".join(repr(item) for item in self) + "}k:" + str(self.k)

    def __ior__(self, other):
        if not isinstance(other, TerminalStrings):
            raise TypeError(f"Expected TerminalStrings, got {other}")
        assert other.k <= self.k
        self._items |= other._items
        return self
