from itertools import islice
from typing import Iterable

from more_itertools import partition
from typeguard import typechecked

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


class TerminalSequence(tuple[Terminal, ...]):
    k: int

    def __new__(cls, terminals: Iterable[Terminal], k: int):
        self = tuple.__new__(TerminalSequence, islice(terminals, k))  # type: ignore
        self.k = k
        return self

    def is_complete(self, k: int):
        return not self.is_eps() and len(self) >= k or (self and self[-1] is EOF)

    @staticmethod
    def eps():
        return TerminalSequence([EMPTY], 1)

    @staticmethod
    def eof():
        return TerminalSequence([EOF], 1)

    def is_eps(self):
        return len(self) == 1 and self[0] is EMPTY

    # Concatenates two collections with respect to the rules of k-concatenation
    @typechecked
    def k_concat(self, other: "TerminalSequence", k: int) -> "TerminalSequence":
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
            return TerminalSequence(terminals, k)

        my_k_len = k_length(terminals, k)
        to_take = k_length(other, k - my_k_len)
        terminals.extend(other[:to_take])
        return TerminalSequence(terminals, k)

    def __repr__(self):
        return "".join(repr(item) for item in self)

    def __str__(self):
        return "".join(str(item) for item in self)


class TerminalSequenceSet(set[TerminalSequence]):
    def __init__(self, items: Iterable[TerminalSequence], k: int):
        super().__init__()
        self.k = k
        for item in items:
            self.add(item)

    @staticmethod
    def intersection(iterable: Iterable["TerminalSequenceSet"]):
        ts_sets = tuple(iterable)
        assert ts_sets
        first = ts_sets[0]
        assert all(
            isinstance(ts_set, TerminalSequenceSet) and ts_set.k == first.k
            for ts_set in ts_sets
        )
        return TerminalSequenceSet(
            set.intersection(*(ts_set for ts_set in ts_sets)), first.k
        )

    @staticmethod
    def of(terminal_string: TerminalSequence, k: int):
        return TerminalSequenceSet([terminal_string], k)

    @staticmethod
    def eps(k):
        assert k >= 1
        return TerminalSequenceSet([TerminalSequence.eps()], k)

    @staticmethod
    def eof(k):
        assert k >= 1
        return TerminalSequenceSet([TerminalSequence.eof()], k)

    @staticmethod
    def empty(k):
        assert k >= 0
        return TerminalSequenceSet([], k)

    def increment_k(self, k: int) -> "TerminalSequenceSet":
        assert k >= self.k
        return TerminalSequenceSet(self, k)

    def is_complete(self):
        return all(item.is_complete(self.k) for item in self)

    def k_concat(self, other_ts_set: "TerminalSequenceSet", k) -> "TerminalSequenceSet":
        if not self.is_complete():
            incomplete, complete = partition(lambda x: x.is_complete(k), self)
            ts_set = TerminalSequenceSet(complete, k)
            for ts in incomplete:
                for other_ts in other_ts_set:
                    ts_set.add(ts.k_concat(other_ts, k))
            return ts_set
        return self

    @typechecked
    def add(self, value: TerminalSequence) -> None:
        assert value.k <= self.k
        super().add(value)

    @typechecked
    def union(self, values: "TerminalSequenceSet") -> "TerminalSequenceSet":
        assert (value.k <= self.k for value in values)
        return TerminalSequenceSet(self | values, self.k)

    @typechecked
    def __or__(self, other: "TerminalSequenceSet"):
        assert other.k <= self.k
        return self.union(other)

    def discard(self, value: TerminalSequence) -> None:
        raise NotImplementedError

    def __repr__(self):
        return "{" + ", ".join(repr(item) for item in self) + "}k:" + str(self.k)

    @typechecked
    def __ior__(self, other: "TerminalSequenceSet"):
        assert other.k <= self.k
        self.update(other)
        return self
