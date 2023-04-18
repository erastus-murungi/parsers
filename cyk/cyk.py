from collections import defaultdict
from itertools import count, product
from typing import Iterator

from more_itertools import one

from grammar import EMPTY, EOF, Expansion, Grammar, NonTerminal, Symbol, Terminal
from parsers.parser import ParseTree
from utils import Token, Tokenizer
from utils.dot import draw_tree
from utils.grammars import GRAMMAR3

Span = tuple[int, int]
Pointer = int | NonTerminal | tuple[int, Symbol, Symbol]
PointerTable = dict[Span, dict[NonTerminal, set[Pointer]]]
CYKResultsTable = dict[Span, set[NonTerminal]]
TodoList = dict[NonTerminal, list[Expansion]]

TERM = "_TERM"
SPLIT = "_SPLIT"
counter = count(0)


def rename_solitary_terminals(
    todo: TodoList, origin: NonTerminal, expansion: Expansion
):
    renamed: list[Symbol] = []
    for symbol in expansion:
        if isinstance(symbol, Terminal):
            non_terminal = NonTerminal(f"{symbol.name}_{next(counter)}{TERM}")
            todo[non_terminal].append(Expansion().append(symbol))
            renamed.append(non_terminal)
        else:
            renamed.append(symbol)
    todo[origin].append(Expansion(renamed))
    todo[origin].remove(expansion)


def split_expansion(todo: TodoList, origin: NonTerminal, expansion: Expansion):
    new_symbol = NonTerminal(f"{origin.name}_{next(counter)}{SPLIT}")
    todo[origin].append(Expansion([new_symbol, expansion[-1]]))
    todo[new_symbol].append(Expansion(expansion[:-1]))


def to_cnf_with_unit_productions(grammar: Grammar) -> Grammar:
    finished = Grammar.Builder(start=grammar.orig_start)
    remaining = grammar.get_mutable_copy()

    while remaining:
        origin, expansions = remaining.popitem()

        for expansion in expansions:
            if EMPTY in expansion:
                raise ValueError(f"CYK does not support empty expansions")

            match expansion:
                case (Symbol(),) | (NonTerminal(), NonTerminal()):
                    finished.add_expansion_no_check(origin, expansion)

                case (Symbol(), Symbol()):
                    rename_solitary_terminals(remaining, origin, expansion)

                case _ if len(expansion) > 2:
                    split_expansion(remaining, origin, expansion)

                case _:
                    raise RuntimeError(f"invalid expansion {expansion}")

    return finished.build()


def yield_trees(
    cnf_grammar: Grammar,
    pointers: PointerTable,
    words: list[Token],
) -> Iterator[ParseTree]:
    assert EOF.matches(words[-1])

    def yield_trees_impl(current: NonTerminal, span: Span) -> Iterator[ParseTree]:
        for pointer in pointers[span][current]:
            match pointer:
                case NonTerminal():
                    for child in yield_trees_impl(pointer, span):
                        yield ParseTree(current, [child])
                case int(pos):
                    yield ParseTree(current, [words[pos]])
                case (int(mid), NonTerminal() as left, NonTerminal() as right):
                    start, end = span
                    yield from (
                        ParseTree(current, list(children))
                        for children in product(
                            yield_trees_impl(left, (start, mid - 1)),
                            yield_trees_impl(right, (mid, end)),
                        )
                    )
                case _:
                    raise ValueError(f"invalid pointer {pointer}")

    yield from yield_trees_impl(cnf_grammar.orig_start, (0, len(words) - 2))


def cyk_parse(
    grammar: Grammar, words: list[Token]
) -> tuple[Grammar, CYKResultsTable, PointerTable]:
    table: CYKResultsTable = defaultdict(set[NonTerminal])
    pointers: PointerTable = defaultdict(lambda: defaultdict(set[Pointer]))
    cnf_grammar: Grammar = to_cnf_with_unit_productions(grammar)

    reversed_grammar = defaultdict(set)
    for root, expansions in cnf_grammar.items():
        for expansion in expansions:
            reversed_grammar[expansion].add(root)

    def add_unaries(span: Span):
        seen = set()
        heads_todo = table[span].copy()
        while heads_todo:
            head = heads_todo.pop()
            if head in seen:
                continue
            seen.add(head)
            for root in reversed_grammar[(head,)]:
                table[span] |= {root}
                pointers[span][root] |= {head}
                heads_todo |= {root}

    for col, word in enumerate(words):
        for root, expansions in cnf_grammar.items():
            for expansion in expansions:
                if (
                    len(expansion) == 1
                    and isinstance(expansion[0], Terminal)
                    and expansion[0].matches(word)
                ):
                    table[(col, col)] |= {root}
                    pointers[(col, col)][root] |= {col}
        add_unaries((col, col))

    for length in range(2, len(words) + 1):
        for col in range(len(words) - length + 1):
            for mid in range(col + 1, col + length):
                for children in product(
                    table[(col, mid - 1)], table[(mid, col + length - 1)]
                ):
                    span = (col, col + length - 1)
                    table[span] |= reversed_grammar[children]
                    for root in reversed_grammar[children]:
                        pointers[span][root] |= {(mid, *children)}
            add_unaries((col, col + length - 1))

    return cnf_grammar, table, pointers


def revert_cnf(parse_tree: ParseTree | Token) -> ParseTree | Token:
    if isinstance(parse_tree, Token):
        return parse_tree
    # * TERM: Eliminates rules with only one terminal symbol on their right-hand-side.
    if parse_tree.id.name.endswith(TERM):
        return one(parse_tree.expansion)
    # * BIN: Eliminates rules with more than 2 symbols on their right-hand-side.
    children = []
    for child in map(revert_cnf, parse_tree.expansion):
        if isinstance(child, ParseTree) and child.id.name.endswith(SPLIT):
            children.extend(child.expansion)
        else:
            children.append(child)
    # * UNIT: Eliminates non-terminal unit rules
    return ParseTree(parse_tree.id, children)


if __name__ == "__main__":
    from rich import print as rprint
    from rich.pretty import pretty_repr

    from utils.parse_grammar import parse_grammar

    g = parse_grammar(GRAMMAR3[1], GRAMMAR3[0])
    rprint(pretty_repr(g))
    print()
    tks = Tokenizer("book the flight through Houston", {}).get_tokens_no_whitespace()

    rprint(pretty_repr(tks))

    cnf_g, tab, sp = cyk_parse(g, tks)
    rprint(pretty_repr(tab))
    rprint(pretty_repr(sp))
    # rprint(pretty_repr(list(yield_trees(cg.start, split, tab, tks))))
    for i, t in enumerate(yield_trees(cnf_g, sp, tks)):
        draw_tree(revert_cnf(t), f"tree_cyk_{i}.pdf")

    # print(len(list(yield_trees(cnf_g, g.start, sp, tab, tks))))
