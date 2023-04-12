import subprocess
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Generic, Hashable, Iterable, Protocol, TypeVar, runtime_checkable

from prettytable import PrettyTable

from grammar import CFG
from grammar.core import NonTerminal, Symbol

FILENAME = "./graphs/state_graph"
DOT_FILEPATH = FILENAME + "." + "dot"
GRAPH_TYPE = "pdf"
OUTPUT_FILENAME = FILENAME + "." + GRAPH_TYPE


@runtime_checkable
class Completable(Protocol, Hashable):
    def completed(self) -> bool:
        ...


T = TypeVar("T", bound=Completable)


class State(list[T]):
    ids: dict["State", int] = defaultdict(count(1).__next__)

    def __init__(self, *items, cls: type[T]):
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
        return State.ids[self]

    def append(self, completable: T) -> None:
        if not isinstance(completable, Completable):
            raise TypeError(f"Expected Completable, got {type(completable)}")
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
        return hash(frozenset(self))

    def __eq__(self, other):
        return frozenset(self) == frozenset(other)

    def __str__(self):
        return "\n".join(str(item) for item in self)


class Action(ABC):
    pass


@dataclass(frozen=True, slots=True)
class Reduce(Action):
    lhs: NonTerminal
    length: int

    def __str__(self):
        return f"Reduce({self.length})"


@dataclass(frozen=True, slots=True)
class Goto(Action, Generic[T]):
    state: State[T]

    def __str__(self):
        return f"Goto({self.state!s})"


@dataclass(frozen=True, slots=True)
class Accept(Action):
    pass


@dataclass(frozen=True, slots=True)
class Shift(Action, Generic[T]):
    state: State[T]

    def __str__(self):
        return f"Shift(\n{self.state!s}\n)"


class LRTable(dict[tuple[State[T], str], Action], ABC):
    def __init__(self, grammar: CFG, *, reduce: bool = True):
        super().__init__()
        self.grammar = grammar
        self.states: list[State[T]] = []
        self.reduce = reduce
        self.accept = None
        self.construct()

    def __hash__(self):
        return id(self)

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
                    edges.append(
                        f'    {hash(str(start))}:from_false -> accept:from_node [arrowhead=vee] [label="$"] '
                    )
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
                case Reduce(name, _):
                    edges.append(
                        f'    {hash(str(start))}:from_node -> {name!s}:from_false [arrowhead=vee, label="reduce [{edge_label}]"] '
                    )

        graph.extend(edges)
        graph.extend(nodes)
        graph.append(graph_epilogue())

        with open(DOT_FILEPATH, "w") as f:
            f.write("\n".join(graph))

        create_graph_pdf()

    def to_pretty_table(self) -> str:
        syms: list[str] = (
            ["State"]
            + [terminal.id for terminal in self.grammar.terminals]
            + [terminal.id for terminal in self.grammar.non_terminals]
        )
        pretty_table = PrettyTable()
        pretty_table.field_names = syms
        hashmap: dict[State[T], dict[str, Action]] = defaultdict(dict)

        for (state, edge_label), action in self.items():
            hashmap[state][edge_label] = action

        rows: list[list[int | str]] = []
        for state, edge_label2action in hashmap.items():
            row: list[int | str] = [state.id]
            for sym in syms[1:]:
                match edge_label2action.get(sym, None):
                    case Goto(state):
                        row.append(f"goto {state.id}")
                    case Shift(state):
                        row.append(f"shift {state.id}")
                    case Reduce(name, length):
                        row.append(f"reduce {name!s}({length})")
                    case Accept():
                        row.append("accept")
                    case _:
                        row.append("")
            rows.append(row)

        rows.sort(key=lambda row: row[0])
        pretty_table.add_rows(rows)

        return pretty_table.get_string()
