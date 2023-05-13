from itertools import islice
from typing import Iterable, cast, Self

from more_itertools import first, one, partition, split_at, take
from typeguard import typechecked

from grammar import EMPTY, EOF, Expansion, NonTerminal, Terminal


def get_k_length(terminals: Iterable[Terminal], k: int) -> int:
    # Returns the k-length, i.e. the number of symbols that contributes to lookahead sizes
    k_len = 0
    for terminal in take(k, terminals):
        k_len += 1
        if terminal is EOF:
            break
    return k_len


class TerminalSequence(tuple[Terminal, ...]):
    def __new__(cls, terminals: Iterable[Terminal], k: int):
        return tuple.__new__(TerminalSequence, islice(terminals, k))  # type: ignore

    def complete(self, k: int) -> bool:
        assert len(self) <= k
        return not self.is_eps() and len(self) == k or (self and self[-1] is EOF)

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

        if self.complete(k):
            # k: w would be the same as k: (w + x)
            return TerminalSequence(terminals, k)

        to_take = get_k_length(other, k - get_k_length(terminals, k))
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
            assert get_k_length(item, k) <= k
            self.add(item)

    @staticmethod
    @typechecked
    def inter(
        ts_sets: tuple["TerminalSequenceSet", ...], k: int
    ) -> "TerminalSequenceSet":
        assert all(ts_set.k == k for ts_set in ts_sets)
        return TerminalSequenceSet(set.intersection(*ts_sets), k)

    @staticmethod
    def of(it: Iterable[Terminal], k: int) -> "TerminalSequenceSet":
        return TerminalSequenceSet([TerminalSequence(it, k)], k)

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

    @typechecked
    def k_concat(self, other_ts_set: "TerminalSequenceSet") -> "TerminalSequenceSet":
        return (
            self
            if all(item.complete(self.k) for item in self)
            else self._k_concat_with_incomplete(other_ts_set)
        )

    def _k_concat_with_incomplete(self, other: Iterable[TerminalSequence]) -> Self:
        incomplete, complete_tss = partition(lambda x: x.complete(self.k), self)
        complete = TerminalSequenceSet(complete_tss, self.k)
        for ts in incomplete:
            for other_ts in other:
                complete.add(ts.k_concat(other_ts, self.k))
        return complete

    def discard(self, element) -> None:
        raise NotImplementedError(
            "we are not meant to remove elements from this set, yet"
        )

    def __repr__(self):
        return f"Set({super().__repr__()}, k={self.k})"

    @typechecked
    def _union(self, other: "TerminalSequenceSet") -> "TerminalSequenceSet":
        assert self.k == other.k
        return TerminalSequenceSet(set.union(self, other), self.k)

    @typechecked
    def _update(self, other: "TerminalSequenceSet") -> "TerminalSequenceSet":
        assert self.k == other.k
        self.update(other)
        return self

    def __or__(self, other):
        return self._union(other)

    def __ior__(self, other):
        return self._update(other)


Part = TerminalSequenceSet | NonTerminal


@typechecked
def gen_parts(expansion: Expansion, k: int) -> list[Part]:
    return [
        cast(NonTerminal, one(symbols))
        if isinstance(first(symbols), NonTerminal)
        else TerminalSequenceSet.of(cast(list[Terminal], symbols), k)
        for symbols in split_at(
            expansion,
            pred=lambda symbol: isinstance(symbol, NonTerminal),
            keep_separator=True,
        )
        if symbols
    ]
