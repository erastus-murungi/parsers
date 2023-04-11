import subprocess
import sys
from abc import ABC
from dataclasses import dataclass
from typing import NamedTuple, Iterable

from cfg import CFG
from core import NonTerminal, Rule, EMPTY, EOF, Symbol, Terminal
from parse_grammar import parse_grammar

FILENAME = "ast"
AST_DOT_FILEPATH = FILENAME + "." + "dot"
AST_GRAPH_TYPE = "pdf"
AST_OUTPUT_FILENAME = FILENAME + "." + AST_GRAPH_TYPE


class LR0Item(NamedTuple):
    name: NonTerminal
    dot: int
    rule: Rule

    def __repr__(self):
        return (
            f"{self.name!r} -> {' '.join(repr(sym) for sym in self.rule[:self.dot])}"
            f" . "
            f"{' '.join(repr(sym) for sym in self.rule[self.dot:])}"
        )

    def advance(self):
        return LR0Item(self.name, self.dot + 1, self.rule)

    def is_completed(self):
        return self.dot >= len(self.rule)


class LR0State(list[LR0Item]):
    def __init__(self, *items):
        assert all(
            isinstance(item, LR0Item) for item in items
        ), "All items must be EarleyItem"
        super().__init__(set(items))

    def append(self, earley_item: LR0Item) -> None:
        if not isinstance(earley_item, LR0Item):
            raise TypeError(f"Expected EarleyItem, got {type(earley_item)}")
        if earley_item not in self:
            super().append(earley_item)

    def extend(self, earley_items: Iterable[LR0Item]) -> None:
        for earley_item in earley_items:
            self.append(earley_item)

    def yield_finished(self):
        for item in self:
            if item.is_completed():
                yield item

    def yield_unfinished(self):
        for item in self:
            if not item.is_completed():
                yield item

    def copy(self) -> "LR0State":
        return LR0State(*self)

    def __hash__(self):
        return hash(tuple(self))


class Action(ABC):
    pass


@dataclass(frozen=True, slots=True)
class Reduce(Action):
    lhs: NonTerminal
    rule: Rule


@dataclass(frozen=True, slots=True)
class Goto(Action):
    state: LR0State


@dataclass(frozen=True, slots=True)
class Accept(Action):
    pass


@dataclass(frozen=True, slots=True)
class Shift(Action):
    state: LR0State


class LR0ParsingTable(dict[tuple[LR0State, str], Action]):
    def __init__(self, grammar: CFG):
        super().__init__()
        self.grammar = grammar
        self.states: list[LR0State] = []
        self.construct()

    def closure(self, state: LR0State):
        # Closure adds more items to a set of items when
        # there is a dot to the left of a non-terminal;
        # The LR(0) closure step adds all items of the form B → •γ to a state
        # whenever the state contains an item A → α • Bβ.

        new_items = state.copy()
        # for all items of the form A → α • Bβ
        changing = True
        while changing:
            changing = False
            for _, dot, rule in new_items.yield_unfinished():
                b = rule[dot]  # α • Bβ
                if isinstance(b, NonTerminal):
                    # add all items of the form B → •γ
                    initial_size = len(new_items)
                    new_items.extend(LR0Item(b, 0, gamma) for gamma in self.grammar[b])
                    changing = len(new_items) != initial_size
        return new_items

    def goto(self, state: LR0State, sym: Symbol) -> LR0State:
        # Goto is the transition function of the LR(0) automaton.
        # It takes a state and a symbol and returns the state that
        # results from shifting the dot over the symbol.
        # If the symbol is a terminal, then the state is unchanged.
        # If the symbol is a non-terminal, then the state is the closure
        # of the state resulting from shifting the dot over the symbol.
        assert sym is not EMPTY and sym is not EOF
        new_items = LR0State()
        for item in state:
            if item.is_completed():
                continue
            if item.rule[item.dot] == sym:
                new_items.append(item.advance())
        return self.closure(new_items)

    def get_initial_kernel(self):
        return LR0State(
            LR0Item(
                self.grammar.start_symbol,
                0,
                self.grammar[self.grammar.start_symbol][0].append_marker(EOF),
            )
        )

    def construct(self):
        states = {self.closure(self.get_initial_kernel()): None}
        changing = True

        while changing:
            changing = False
            for state in states.copy():
                for _, dot, rule in state.yield_unfinished():
                    symbol = rule[dot]
                    if symbol is EOF:
                        # accept action
                        self[(state, symbol.id)] = Accept()
                    else:
                        # shift action
                        target = self.goto(state, symbol)
                        assert target
                        if target not in states:
                            changing = True
                            states[target] = None
                        action = (
                            Shift(target)
                            if isinstance(symbol, Terminal)
                            else Goto(target)
                        )
                        if action != self.get((state, symbol.id), None):
                            self[(state, symbol.id)] = action
                            changing = True

        # compute reduce actions
        for state in states:
            for item in state.yield_finished():
                for symbol in self.grammar.terminals:
                    if (state, symbol) not in self:
                        self[(state, symbol.id)] = Reduce(item.name, item.rule)
                    else:
                        raise ValueError(
                            f"Encountered reduce/reduce conflict {state!s} {symbol!s}"
                        )
        self.states = list(states)

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
            dot_filepath=AST_DOT_FILEPATH,
            output_filepath=AST_OUTPUT_FILENAME,
            output_filetype=AST_GRAPH_TYPE,
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
            subprocess.run(["rm", AST_DOT_FILEPATH])

        def str_state(state: LR0State):
            return escape(
                "\n".join(
                    f"{item.name!s} -> {' '.join(str(sym) for sym in item.rule[:item.dot])}"
                    f" . "
                    f"{' '.join(str(sym) for sym in item.rule[item.dot:])}"
                    for item in state
                )
            )

        graph = [graph_prologue()]
        edges = []
        nodes = []
        seen = set()
        for (start, edge_label), action in self.items():
            if start not in seen:
                nodes.append(
                    f"   {hash(str(start))} [shape=record, style=filled, fillcolor=black, "
                    f'fontcolor=white, label="{str_state(start)}"];'
                )
            seen.add(start)
            match action:
                case Accept():
                    pass
                case Shift(state):
                    if state not in seen:
                        nodes.append(
                            f"   {hash(str(state))} [shape=record, style=filled, "
                            f'fillcolor=black, fontcolor=white, label="{str_state(state)}"];'
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
                            f'fillcolor=black, fontcolor=white, label="{str_state(state)}"];'
                        )
                    edges.append(
                        f"    {hash(str(start))}:from_false -> {hash(str(state))}:from_node "
                        f'[label="goto [{escape(edge_label)}]"];'
                    )
                    seen.add(state)
                case Reduce(name, rule):
                    pass

        graph.extend(edges)
        graph.extend(nodes)
        graph.append(graph_epilogue())
        graph.append(graph_epilogue())

        with open(AST_DOT_FILEPATH, "w") as f:
            f.write("\n".join(graph))

        create_graph_pdf()


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr

    table = {
        "x": "x",
        "(": "(",
        ")": ")",
        ",": ",",
    }
    g = """
            <S'>
            <S'> -> <S>
            <S> -> (<L>)
            <L> -> <S>
            <S> -> x
            <L> -> <L>,<S>
    """

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))

    print_rich(pretty_repr(LR0ParsingTable(cfg)))
