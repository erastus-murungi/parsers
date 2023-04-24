import subprocess
import sys
from typing import Iterator

from grammar import Terminal
from lr import Accept, Goto, LRTable, Reduce, Shift
from parsers.parser import AST, ParseTree

DIR = "./graphs/"
DOT_FILENAME = "tree.dot"
GRAPH_TYPE = "pdf"


def yield_edges(
    root,
) -> Iterator[tuple[AST, Terminal]] | Iterator[tuple[ParseTree, Terminal]]:
    if isinstance(root, ParseTree):
        yield from yield_edges_parse_tree(root)
    else:
        yield from yield_edges_ast(root)


def yield_edges_ast(root) -> Iterator[tuple[AST, Terminal]]:
    for child in root["expansion"]:
        if isinstance(child, Terminal):
            yield root, child
        else:
            yield root, child
            yield from yield_edges_ast(child)


def yield_edges_parse_tree(root) -> Iterator[tuple[ParseTree, Terminal]]:
    for child in root.expansion:
        if isinstance(child, Terminal):
            yield root, child
        else:
            yield root, child
            yield from yield_edges_parse_tree(child)


def graph_prologue() -> str:
    return (
        'digraph G {  graph [fontname = "Courier New", engine="sfdp"];\n'
        + ' node [fontname = "Courier", style = rounded];\n'
        + ' edge [fontname = "Courier"];'
    )


def graph_epilogue() -> str:
    return "}"


def escape(s: str) -> str:
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
    graph,
    output_filename,
    dot_filename=DOT_FILENAME,
    output_filetype=GRAPH_TYPE,
):
    dot_exec_filepath = (
        "/usr/local/bin/dot" if sys.platform == "darwin" else "/usr/bin/dot"
    )

    output_filepath = DIR + output_filename
    dot_filepath = DIR + dot_filename

    with open(dot_filepath, "w+") as f:
        f.write("\n".join(graph))

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
    # subprocess.run(["rm", output_filepath])


def draw_tree(root: ParseTree | AST | Terminal, output_filename: str = "tree.pdf"):
    graph = [graph_prologue()]
    edges = []
    nodes = []

    def get_label_non_token(node):
        if isinstance(node, ParseTree):
            return node.id.name
        else:
            return node["id"]

    for src, dst in yield_edges(root):
        if isinstance(src, Terminal):
            nodes.append(
                f"   {id(src)} [shape=doublecircle, style=filled, fillcolor=white, "
                f'fontcolor=black, label="{escape(src.lexeme)}"];'
            )
        else:
            nodes.append(
                f"   {id(src)} [shape=record, style=filled, fillcolor=black, "
                f'fontcolor=white, label="{escape(get_label_non_token(src))}"];'
            )
        if isinstance(dst, Terminal):
            nodes.append(
                f"   {id(dst)} [shape=doublecircle, style=filled, fillcolor=white, "
                f'fontcolor=black, label="{escape(dst.lexeme)}"];'
            )
        else:
            nodes.append(
                f"   {id(dst)} [shape=record, style=filled, fillcolor=black, "
                f'fontcolor=white, label="{get_label_non_token(dst)}"];'
            )
        edges.append(
            f"{(id(src))}:from_false -> {(id(dst))}:from_node [arrowhead=vee] "
        )

    graph.extend(edges)
    graph.extend(nodes)
    graph.append(graph_epilogue())

    create_graph_pdf(graph, output_filename=output_filename)


def draw_state_graph(table: LRTable, output_filename="state_graph.pdf"):
    graph = [graph_prologue()]
    edges = []
    nodes = []
    seen = set()

    edges.append(
        f"    start:from_false -> {hash(str(table.states[0]))}:from_node [arrowhead=vee] "
    )
    for (start, edge_label), action in table.items():
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
                    f"    {hash(str(start))}:from_node -> {name!s}:from_false "
                    f'[arrowhead=vee, label="reduce [{edge_label}]"] '
                )

    graph.extend(edges)
    graph.extend(nodes)
    graph.append(graph_epilogue())

    create_graph_pdf(graph, output_filename)
