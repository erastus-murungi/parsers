from abc import ABC, abstractmethod
from itertools import product
from typing import Iterator, NamedTuple, TypedDict, Union, cast

from rich import print as print_rich
from rich.pretty import pretty_repr
from rich.traceback import install

from cfg import CFG
from core import EMPTY, EOF, NonTerminal, Terminal
from earley import EarleyItem, gen_early_sets
from parse_grammar import parse_grammar
from tokenizer import Token, Tokenizer

install(show_locals=True)


class AST(TypedDict):
    id: str
    expansion: list["AST"]


class ParseTree(NamedTuple):
    id: NonTerminal
    expansion: list[Union["ParseTree", Token]]

    def collapse(self) -> AST:
        expansion = []
        for child in self.expansion:
            if isinstance(child, Token):
                expansion.append(child.lexeme)
            else:
                child_collapse = child.collapse()
                assert "id" in child_collapse
                if child_collapse:
                    if len(child_collapse["expansion"]) == 1:
                        expansion.extend(child_collapse["expansion"])
                    else:
                        expansion.append(child_collapse)
        return {"id": self.id.id, "expansion": expansion}


class Parser(ABC):
    @abstractmethod
    def parse(self, tokens: list[Token]) -> Iterator[ParseTree]:
        """Parse a list of tokens into a parse tree"""
        ...


class LL1Parser(Parser):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    def parse(self, tokens: list[Token]) -> Iterator[ParseTree]:
        parsing_table = self.grammar.build_ll1_parsing_table()
        root = ParseTree(self.grammar.start_symbol, [])
        stack, token_index = [
            (EOF, root),
            (self.grammar.start_symbol, root),
        ], 0

        while stack:
            symbol, root = stack.pop()
            token = tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
                    root.expansion.append(token)
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.id} but got {token}")
            else:
                non_terminal = cast(NonTerminal, symbol)
                if (rule := parsing_table.get((non_terminal, token.id))) is not None:
                    nodes = [ParseTree(sym, []) for sym in rule]
                    root.expansion.extend(nodes)
                    stack.extend(reversed(list(zip(rule, nodes))))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.id for terminal in self.grammar.first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert token_index >= len(tokens)
        print_rich(pretty_repr(root.collapse()))
        return root


class EarleyParser(Parser):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    def parse(self, tokens: list[Token]) -> Iterator[ParseTree]:
        """Parse a list of tokens into a parse tree"""

        earley_sets = [
            earley_set.remove_unfinished()
            for earley_set in gen_early_sets(self.grammar, tokens)
        ]

        print_rich(
            pretty_repr({pos: earley_set for pos, earley_set in enumerate(earley_sets)})
        )

        # reverse the earley sets
        parse_forest: list[list[EarleyItem]] = [[] for _ in range(len(tokens))]
        for end_index, earley_set in enumerate(earley_sets):
            for earley_item in earley_set:
                name, dot, start, rule = earley_item
                parse_forest[start].append(EarleyItem(name, dot, end_index, rule))

        def yield_all_paths(
            path: list[EarleyItem | Token],
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
                    if isinstance(item, Token):
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
                    print_rich(pretty_repr(tree.collapse()))
                    yield tree


if __name__ == "__main__":
    # g = """
    #     <program>
    #     <program> -> <expression>
    #     <expression> -> <term> | <term> <add_op> <expression>
    #     <term> -> <factor> | <factor> <mult_op> <term>
    #     <factor> -> <power> | <power> ^ <factor>
    #     <power> -> <number> | ( <expression> )
    #     <number> -> <digit> | <digit> <number>
    #     <add_op> -> + | -
    #     <mult_op> -> * | /
    #     <digit> -> integer | float
    # """

    table = {
        "(": "(",
        ")": ")",
        "-": "-",
        "/": "/",
        "or_literal": "|",
        "?:": "capture",
        "$": "end",
        "^": "start",
        ".": "any",
        "\\d": "digit",
        "\\D": "not_digit",
        "\\s": "space",
        "\\S": "not_space",
        "\\w": "word",
        "\\W": "not_word",
    }

    # g = """
    #     <S>
    #     <S> -> <MaybeOpeningAnchor><Expr><MaybeClosingAnchor>
    #     <Expr> -> <Expr> or_literal <Term> | <Term>
    #     <Term> -> <Factor> <Term> | <Factor>
    #     <Factor> -> <Atom> <Quantifier> | <Atom>
    #     <Atom> -> <Char> | <Group>
    #     <Char> -> <Literal> | <Metachar>
    #     <Quantifier> -> * | + | ? | { integer } | { integer , } | { integer , integer } | { , integer }
    #     <Literal> -> char
    #     <Metachar> -> . | <CharacterClass>
    #     <Group> -> (<MaybeCapture> <Expr> )
    #     <MaybeCapture> -> ?: |
    #     <MaybeOpeningAnchor> -> ^ |
    #     <MaybeClosingAnchor> -> $ |
    #
    #     <CharacterClass> -> \\d | \\D | \\s | \\S | \\w | \\W
    #
    # """
    table = {
        "(": "(",
        ")": ")",
        "+": "+",
        "*": "*",
    }

    g = """
         <S>
         <S> -> <E>
         <E> -> integer
         <E> -> (<E> <Op> <E>)
         <Op> -> + | *
    """

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
    #
    # tks = Tokenizer("^a+(?:ab)c{1,3}$", table).get_tokens_no_whitespace()
    tks = Tokenizer("(1 + 1)", table).get_tokens_no_whitespace()
    print_rich(pretty_repr(tks))

    earley_parser = LL1Parser(g)
    list(earley_parser.parse(tks))
