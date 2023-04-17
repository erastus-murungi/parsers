import re
from collections import defaultdict
from itertools import count, product

from more_itertools import first, only

from grammar import EOF, Expansion, Grammar, NonTerminal, Symbol, Terminal
from parsers.parser import ParseTree
from utils import Token, Tokenizer
from utils.dot import draw_tree
from utils.grammars import GRAMMAR3

Span = tuple[int, int]

TERM = "_TERM_"
SPLIT = "_SPLIT_"


def rename_in_expansion(work_list, origin, expansion, replacer, to_replace):
    if to_replace in expansion:
        new_expansion = [replacer if sym == to_replace else sym for sym in expansion]
        work_list[origin].append(Expansion(new_expansion))
        work_list[origin].remove(expansion)


def rename(
    work_list: dict[NonTerminal, list], exp: Expansion, replacer: NonTerminal
) -> None:
    to_replace = first(exp)
    for origin, expansions in work_list.items():
        for expansion in expansions:
            rename_in_expansion(work_list, origin, expansion, replacer, to_replace)
    try:
        if work_list[to_replace]:
            work_list[replacer].extend(work_list[to_replace])
        work_list[replacer].remove(exp)
    except ValueError:
        pass


def to_cnf(grammar: Grammar) -> Grammar:
    finished = defaultdict(list)
    worklist = defaultdict(list)
    for origin, expansions in grammar.items():
        for expansion in expansions:
            worklist[origin].append(expansion)

    counter = count(0)
    while worklist:
        origin, expansions = worklist.popitem()
        for expansion in expansions:
            match expansion:
                case (Terminal(),) | (NonTerminal(), NonTerminal()):
                    finished[origin].append(expansion)
                case (Terminal(), NonTerminal()):
                    rename_in_expansion(
                        worklist,
                        origin,
                        expansion,
                        NonTerminal(f"{expansion[0].name}{TERM}{next(counter)}"),
                        expansion[0],
                    )
                case (NonTerminal(), Terminal()):
                    rename_in_expansion(
                        worklist,
                        origin,
                        expansion,
                        NonTerminal(f"{expansion[1].name}{TERM}{next(counter)}"),
                        expansion[1],
                    )
                case (NonTerminal(),):
                    if origin == grammar.start:
                        nt = NonTerminal(f"{TERM}EOF")
                        finished[origin].append(expansion.append(nt))
                        finished[nt].append(Expansion([EOF]))
                    else:
                        rename(finished, expansion, origin)
                        rename(worklist, expansion, origin)
                case (*prefix, last) if len(prefix) > 1:
                    new_symbol = NonTerminal(f"{origin.name}{SPLIT}{next(counter)}")
                    worklist[origin].append(Expansion([new_symbol, last]))
                    worklist[new_symbol].append(Expansion(prefix))
                case _:
                    raise ValueError(f"invalid expansion {expansion}")
    cnf = Grammar.Builder(grammar.start.name)
    for sym, expansions in finished.items():
        cnf.add_definition(sym, set(expansions))
    return cnf.build()


def is_cnf(grammar: Grammar) -> bool:
    for expansions in grammar.values():
        for expansion in expansions:
            if len(expansion) > 2:
                return False
            if len(expansion) == 2 and not all(isinstance(sym, NonTerminal) for sym in expansion):
                return False
    return True


def yield_trees(
    root: Symbol,
    split_table: dict[Span, set[int]],
    table: dict[Span, set[NonTerminal]],
    words: list[Token],
) -> ParseTree:
    def yield_trees_impl(current: Symbol, span: Span) -> ParseTree:
        start, end = span
        if start == end:
            yield ParseTree(current, [words[start]])
        else:
            for split in split_table[span]:
                left_span, right_span = (start, split - 1), (split, end)
                for left, right in product(table[left_span], table[right_span]):
                    for children in product(
                        yield_trees_impl(left, left_span),
                        yield_trees_impl(right, right_span),
                    ):
                        yield ParseTree(current, list(children))

    yield from yield_trees_impl(root, (0, len(words) - 1))


def cyk_parse(grammar: Grammar, words: list[Token]):
    table = defaultdict(set[NonTerminal])
    split_table = defaultdict(set[int])

    reversed_grammar = defaultdict(set)
    for origin, expansions in grammar.items():
        for expansion in expansions:
            reversed_grammar[expansion].add(origin)

    for start, word in enumerate(words):
        for origin, expansions in grammar.items():
            for expansion in expansions:
                match expansion:
                    case (Terminal() as terminal,) if terminal.matches(word):
                        table[(start, start)].add(origin)

    for length in range(2, len(words) + 1):
        for start in range(len(words) - length + 1):
            for split in range(start + 1, start + length):
                left_span = (start, split - 1)
                right_span = (split, start + length - 1)
                for rule in product(table[left_span], table[right_span]):
                    table[(start, start + length - 1)] |= set(reversed_grammar[rule])
                    split_table[(start, start + length - 1)].add(split)

    return table, split_table


def revert_cnf(parse_tree: ParseTree | Token) -> ParseTree | Token:
    if isinstance(parse_tree, Token):
        return parse_tree
    # * TERM: Eliminates rules with only one terminal symbol on their right-hand-side.
    if re.match(f'{TERM}\\d+$', parse_tree.id.name) is not None:
        return only(parse_tree.expansion)
    # * BIN: Eliminates rules with more than 2 symbols on their right-hand-side.
    children = []
    for child in map(revert_cnf, parse_tree.expansion):
        if isinstance(child, ParseTree) and re.match(f'{SPLIT}\\d+$', child.id.name):
            children.extend(child.expansion)
        else:
            children.append(child)
    # * UNIT: Eliminates non-terminal unit rules
    return ParseTree(parse_tree.id, children)


if __name__ == "__main__":
    from rich import print as rprint
    from rich.pretty import pretty_repr

    from utils.parse_grammar import parse_grammar

    t, g = GRAMMAR3
    g = parse_grammar(g, t)
    rprint(pretty_repr(g))
    print()
    rprint(pretty_repr(to_cnf(g)))

    tks = Tokenizer("book the flight through Houston", {}).get_tokens_no_whitespace()

    rprint(pretty_repr(tks))

    cg = to_cnf(g)
    tab, sp = cyk_parse(cg, tks)
    rprint(pretty_repr(tab))
    rprint(pretty_repr(sp))
    # # rprint(pretty_repr(list(yield_trees(cg.start, split, tab, tks))))
    # for i, t in enumerate(yield_trees(cg.start, sp, tab, tks)):
    #     draw_tree(t, output_filename=f"tree_{i}.pdf")
