import subprocess
import sys
from typing import NamedTuple

from cfg import CFG
from core import EMPTY, EOF, NonTerminal, Rule, Symbol, Terminal
from lr_common import Accept, Action, Goto, Reduce, Shift, State

FILENAME = "state_graph"
DOT_FILEPATH = FILENAME + "." + "dot"
GRAPH_TYPE = "pdf"
OUTPUT_FILENAME = FILENAME + "." + GRAPH_TYPE


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

    def completed(self):
        return self.dot >= len(self.rule)


class LR0ParsingTable(dict[tuple[State[LR0Item], str], Action]):
    def __init__(self, grammar: CFG):
        super().__init__()
        self.grammar = grammar
        self.states: list[State[LR0Item]] = []
        self.construct()

    def closure(self, state: State[LR0Item]):
        # Closure adds more items to a set of items when
        # there is a dot to the left of a non-terminal;
        # The LR(0) closure step adds all items of the form B → •γ to a state
        # whenever the state contains an item A → α • Bβ.

        new_items: State[LR0Item] = state.copy()
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

    def goto(self, state: State[LR0Item], sym: Symbol) -> State[LR0Item]:
        # Goto is the transition function of the LR(0) automaton.
        # It takes a state and a symbol and returns the state that
        # results from shifting the dot over the symbol.
        # If the symbol is a terminal, then the state is unchanged.
        # If the symbol is a non-terminal, then the state is the closure
        # of the state resulting from shifting the dot over the symbol.
        assert sym is not EMPTY and sym is not EOF
        kernel = State[LR0Item](cls=LR0Item)
        for item in state.yield_unfinished():
            if item.rule[item.dot] == sym:
                kernel.append(item.advance())
        return self.closure(kernel)

    def get_initial_kernel(self):
        return State[LR0Item](
            LR0Item(
                self.grammar.start_symbol,
                0,
                self.grammar[self.grammar.start_symbol][0].append_marker(EOF),
            ),
            cls=LR0Item,
        )

    def compute_reduce_actions(self):
        for state in self.states:
            for item in state.yield_finished():
                for symbol in self.grammar.terminals:
                    if (state, symbol.id) not in self:
                        self[(state, symbol.id)] = Reduce(item.name, item.rule)
                    else:
                        raise ValueError(
                            f"Encountered shift/reduce conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.id}\n"
                            f"  {self[(state, symbol.id)]} and \n"
                            f"  Reduce({item.name!s} -> {item.rule!s})"
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

        self.states = list(states)
        self.compute_reduce_actions()

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
                case Reduce(name, rule):
                    pass

        graph.extend(edges)
        graph.extend(nodes)
        graph.append(graph_epilogue())

        with open(DOT_FILEPATH, "w") as f:
            f.write("\n".join(graph))

        create_graph_pdf()


class SLRParsingTable(LR0ParsingTable):
    def compute_reduce_actions(self):
        follow_set = self.grammar.follow()
        for state in self.states:
            for item in state.yield_finished():
                for symbol in follow_set[item.name]:
                    if (state, symbol.id) not in self:
                        self[(state, symbol.id)] = Reduce(item.name, item.rule)
                    else:
                        raise ValueError(
                            f"Encountered shift/reduce conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.id}\n"
                            f"  {self[(state, symbol.id)]} and \n"
                            f"  Reduce({item.name!s} -> {item.rule!s})"
                        )


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr

    from parse_grammar import parse_grammar

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
