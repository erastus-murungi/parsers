import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Hashable, Iterable, Protocol, TypeVar, runtime_checkable

from core import NonTerminal, Rule, Symbol

FILENAME = "state_graph"
DOT_FILEPATH = FILENAME + "." + "dot"
GRAPH_TYPE = "pdf"
OUTPUT_FILENAME = FILENAME + "." + GRAPH_TYPE


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


class LRTable(dict[tuple[State[T], str], Action], ABC):
    def __init__(self, grammar):
        super().__init__()
        self.grammar = grammar
        self.states: list[State[T]] = []
        self.construct()

    @abstractmethod
    def closure(self, state: State[T]):
        pass

    @abstractmethod
    def goto(self, state: State[T], sym: Symbol) -> State[T]:
        pass

    @abstractmethod
    def init_kernel(self) -> State[T]:
        pass

    @abstractmethod
    def compute_reduce_actions(self):
        pass

    @abstractmethod
    def construct(self):
        pass

    def draw_with_graphviz(self):
        def graph_prologue():
            return (
                'digraph G {  graph [fontname = "Courier New", engine="sfdp"];\n'
                + ' node [fontname = "Courier", style = rounded];\n'
                + ' edge [fontname = "Courier"];'
            )

        def graph_epilogue():
            return "}"

        def escape(s: str):
            return (
                s.replace("\\", "\\\\")
                .replace("\t", "\\t")
                .replace("\b", "\\b")
                .replace("\r", "\\r")
                .replace("\f", "\\f")
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace("<", "\\<")
                .replace(">", "\\>")
                .replace("\n", "\\l")
                .replace("||", "\\|\\|")
                .replace("[", "\\[")
                .replace("]", "\\]")
                .replace("{", "\\{")
                .replace("}", "\\}")
            )

        def create_graph_pdf(
            dot_filepath=DOT_FILEPATH,
            output_filepath=OUTPUT_FILENAME,
            output_filetype=GRAPH_TYPE,
        ):
            dot_exec_filepath = (
                "/usr/local/bin/dot" if sys.platform == "darwin" else "/usr/bin/dot"
            )
            args = [
                dot_exec_filepath,
                f"-T{output_filetype}",
                f"-Gdpi={96}",
                dot_filepath,
                "-o",
                output_filepath,
            ]
            subprocess.run(args)
            subprocess.run(["open", output_filepath])
            subprocess.run(["rm", DOT_FILEPATH])

        graph = [graph_prologue()]
        edges = []
        nodes = []
        seen = set()

        edges.append(
            f"    start:from_false -> {hash(str(self.states[0]))}:from_node [arrowhead=vee] "
        )
        for (start, edge_label), action in self.items():
            if start not in seen:
                nodes.append(
                    f"   {hash(str(start))} [shape=record, style=filled, fillcolor=black, "
                    f'fontcolor=white, label="{escape(str(start))}"];'
                )
            seen.add(start)
            match action:
                case Accept():
                    pass
                case Shift(state):
                    if state not in seen:
                        nodes.append(
                            f"   {hash(str(state))} [shape=record, style=filled, "
                            f'fillcolor=black, fontcolor=white, label="{escape(str(state))}"];'
                        )
                    edges.append(
                        f"    {hash(str(start))}:from_false -> {hash(str(state))}:from_node "
                        f'[label="shift [{escape(edge_label)}]"];'
                    )
                    seen.add(state)
                case Goto(state):
                    if state not in seen:
                        nodes.append(
                            f"   {hash(str(state))} [shape=record, style=filled, "
                            f'fillcolor=black, fontcolor=white, label="{escape(str(state))}"];'
                        )
                    edges.append(
                        f"    {hash(str(start))}:from_false -> {hash(str(state))}:from_node "
                        f'[label="goto [{escape(edge_label)}]"];'
                    )
                    seen.add(state)
                case Reduce(_, _):
                    pass

        graph.extend(edges)
        graph.extend(nodes)
        graph.append(graph_epilogue())

        with open(DOT_FILEPATH, "w") as f:
            f.write("\n".join(graph))

        create_graph_pdf()
