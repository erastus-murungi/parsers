from abc import ABC, abstractmethod
from itertools import product
from typing import Union, Iterator, NamedTuple

from rich import print as print_rich
from rich.pretty import pretty_repr
from rich.traceback import install

from cfg import CFG
from core import NonTerminal, Terminal
from earley import gen_early_sets, EarleyItem
from parse_grammar import parse_grammar
from tokenizer import Tokenizer

install(show_locals=True)


class ParseTree(NamedTuple):
    id: NonTerminal
    expansion: tuple[Union["ParseTree", Tokenizer.Token], ...]


class Parser(ABC):
    @abstractmethod
    def parse(self, tokens: list[Tokenizer.Token]) -> Iterator[ParseTree]:
        """Parse a list of tokens into a parse tree"""
        ...


class EarleyParser(Parser):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    def parse(self, tokens: list[Tokenizer.Token]) -> Iterator[ParseTree]:
        """Parse a list of tokens into a parse tree"""

        earley_sets = [
            earley_set.remove_unfinished()
            for earley_set in gen_early_sets(self.grammar, tokens)
        ]

        # reverse the earley sets
        parse_forest: list[list[EarleyItem]] = [[] for _ in range(len(tokens))]
        for end_index, earley_set in enumerate(earley_sets):
            for earley_item in earley_set:
                name, dot, start, rule = earley_item
                parse_forest[start].append(EarleyItem(name, dot, end_index, rule))

        def yield_all_paths(
            path: list[EarleyItem | Tokenizer.Token],
            start_index: int,
            left,
            end_index: int,
        ) -> list[EarleyItem]:
            # if we are at a leaf
            if not left:
                if end_index == start_index:
                    yield path
            else:
                current_symbol, *left = left
                if isinstance(current_symbol, Terminal):
                    current_token = tokens[start_index]
                    if current_symbol.matches(current_token):
                        yield from yield_all_paths(
                            path + [current_token],
                            start_index + 1,
                            left,
                            end_index,
                        )
                else:
                    for next_item in parse_forest[start_index]:
                        if next_item.name == current_symbol:
                            yield from yield_all_paths(
                                path + [next_item],
                                next_item.explicit_index,
                                left,
                                end_index,
                            )

        def build_tree(
            earley_item: EarleyItem, start_index: int, end_index: int
        ) -> Iterator[ParseTree]:
            # yield all the trees from the children
            for path in yield_all_paths([], start_index, earley_item.rule, end_index):
                children_possibilities = []
                for item in path:
                    if isinstance(item, Tokenizer.Token):
                        children_possibilities.append([item])
                        start_index += 1
                    else:
                        children_possibilities.append(
                            list(build_tree(item, start_index, item.explicit_index))
                        )
                        start_index = item.explicit_index
                for children in product(*children_possibilities):
                    yield ParseTree(earley_item.name, children)

        n_tokens = len(tokens) - 1  # ignore EOF
        for earley_item in parse_forest[0]:
            if (
                earley_item.explicit_index == n_tokens
                and earley_item.name == self.grammar.start_symbol
            ):
                for tree in build_tree(earley_item, 0, n_tokens):
                    yield tree


if __name__ == "__main__":
    g = """
        <Program>
        <Program>       -> <Sum>
        <Sum>           -> <Sum> <PlusOrMinus> <Product> | <Product>
        <Product>       -> <Product> <MulOrDiv> <Factor> | <Factor>
        <Factor>        -> (<Sum>) | <Number>
        <Number>        -> integer
        <PlusOrMinus>   -> + | -
        <MulOrDiv>      -> * | /
    """

    table = {"(": "(", ")": ")", "+": "+", "-": "-", "*": "*", "/": "/"}

    # g = """
    # <G>
    # <G> -> <A>
    # <A> -> <> | <B>
    # <B> -> <A>
    #
    # """
    # table = {}

    g = parse_grammar(g, table)
    print_rich(pretty_repr(g))

    tks = Tokenizer("1 + (2*3 - 4)", table).get_tokens_no_whitespace()

    earley_parser = EarleyParser(g)
    print_rich(pretty_repr(list(earley_parser.parse(tks))))
