from abc import ABC, abstractmethod
from itertools import product
from typing import Iterator, NamedTuple, Required, TypedDict, Union, cast

from rich import print as print_rich
from rich.pretty import pretty_repr
from rich.traceback import install

from cfg import CFG
from core import EMPTY, EOF, NonTerminal, Symbol, Terminal
from earley import EarleyItem, gen_early_sets
from lr_common import State
from parse_grammar import parse_grammar
from tokenizer import Token, Tokenizer

install(show_locals=True)


class AST(TypedDict):
    id: Required[str]
    expansion: Required[list[Union[str, "AST"]]]


class ParseTree(NamedTuple):
    id: Symbol
    expansion: list[Union["ParseTree", Token]]

    def collapse(self) -> AST:
        expansion: list[AST | str] = []
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
    def parse(self, tokens: list[Token]) -> Iterator[ParseTree] | ParseTree:
        """Parse a list of tokens into a parse tree"""
        ...


class LL1Parser(Parser):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    def parse(self, tokens: list[Token]) -> Iterator[ParseTree] | ParseTree:
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
            State[EarleyItem](*earley_set.yield_finished(), cls=EarleyItem)
            for earley_set in gen_early_sets(self.grammar, tokens)
        ]

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
            path_end_index: int,
        ) -> Iterator[list[EarleyItem | Token]]:
            # if we are at a leaf
            if not left:
                if path_end_index == start_index:
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
                            path_end_index,
                        )
                else:
                    for next_item in parse_forest[start_index]:
                        if next_item.name == current_symbol:
                            yield from yield_all_paths(
                                path + [next_item],
                                next_item.explicit_index,
                                left,
                                path_end_index,
                            )

        def build_tree(
            path_root: EarleyItem, path_start_index: int, path_end_index: int
        ) -> Iterator[ParseTree]:
            # yield all the trees from the children
            for path in yield_all_paths(
                [], path_start_index, path_root.rule, path_end_index
            ):
                item_start_index = path_start_index
                children_possibilities: list[list[Token | ParseTree]] = []
                for item in path:
                    if isinstance(item, Token):
                        children_possibilities.append([item])
                        item_start_index += 1
                    else:
                        children_possibilities.append(
                            list(
                                build_tree(item, item_start_index, item.explicit_index)
                            )
                        )
                        item_start_index = item.explicit_index
                for children in product(*children_possibilities):
                    assert len(children) == len(path_root.rule)
                    yield ParseTree(path_root.name, list(children))

        n_tokens = len(tokens) - 1  # ignore EOF
        for earley_item in parse_forest[0]:
            if (
                earley_item.explicit_index == n_tokens
                and earley_item.name == self.grammar.start_symbol
            ):
                for tree in build_tree(earley_item, 0, n_tokens):
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

    g = """
        <S>
        <S> -> <MaybeOpeningAnchor><Expr><MaybeClosingAnchor>
        <Expr> -> <Expr> or_literal <Term> | <Term>
        <Term> -> <Factor> <Term> | <Factor>
        <Factor> -> <Atom> <Quantifier> | <Atom>
        <Atom> -> <Char> | <Group>
        <Char> -> <Literal> | <Metachar>
        <Quantifier> -> * | + | ? | { integer } | { integer , } | { integer , integer } | { , integer }
        <Literal> -> char
        <Metachar> -> . | <CharacterClass>
        <Group> -> (<MaybeCapture> <Expr> )
        <MaybeCapture> -> ?: |
        <MaybeOpeningAnchor> -> ^ |
        <MaybeClosingAnchor> -> $ |

        <CharacterClass> -> \\d | \\D | \\s | \\S | \\w | \\W

    """
    # table = {
    #     "+": "+",
    #     "-": "-",
    #     "*": "*",
    #     "a": "a",
    # }
    #
    # g = """
    #     <S>
    #     <S> -> <A>
    #     <A> -> <A> + <A> | <A> − <A> | a
    # """

    # table = {
    #     "(": "(",
    #     ")": ")",
    #     "+": "+",
    #     "*": "*",
    # }
    #
    # g = """
    #      <S>
    #      <S> -> <E>
    #      <E> -> integer
    #      <E> -> (<E> <Op> <E>)
    #      <Op> -> + | *
    # """

    # g = """
    # <G>
    # <G> -> <A>
    # <A> -> <> | <B>
    # <B> -> <A>
    #
    # """
    # table = {}

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))

    # tks = Tokenizer("a + a − a", table).get_tokens_no_whitespace()
    #
    tks = Tokenizer("^a+(?:ab)c{1,3}$", table).get_tokens_no_whitespace()
    # tks = Tokenizer("(1 + 1)", table).get_tokens_no_whitespace()
    print_rich(pretty_repr(tks))

    earley_parser = EarleyParser(cfg)
    print_rich(pretty_repr(list(earley_parser.parse(tks))))
