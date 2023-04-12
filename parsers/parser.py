from abc import ABC, abstractmethod
from itertools import product
from typing import Iterator, NamedTuple, Required, TypedDict, Union, cast

from more_itertools import one
from rich import print as print_rich
from rich.pretty import pretty_repr
from rich.traceback import install

from earley import EarleyItem, gen_early_sets
from grammar import CFG, EMPTY, EOF, NonTerminal, Symbol, Terminal
from ll import LL1ParsingTable
from lr import (
    Accept,
    Goto,
    LALR1ParsingTable,
    LR0ParsingTable,
    LR1ParsingTable,
    LRState,
    Reduce,
    Shift,
    SLRParsingTable,
)
from utils import Token, Tokenizer

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
                    if len(child_collapse["expansion"]) <= 1:
                        expansion.extend(child_collapse["expansion"])
                    else:
                        expansion.append(child_collapse)
        return {"id": self.id.name, "expansion": expansion}


class Parser(ABC):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    @abstractmethod
    def parse(self, tokens: list[Token]) -> Iterator[ParseTree] | ParseTree:
        """Parse a list of tokens into a parse tree"""
        ...


class LL1Parser(Parser):
    def parse(self, tokens: list[Token]) -> Iterator[ParseTree] | ParseTree:
        parsing_table = LL1ParsingTable(self.grammar)
        print_rich(pretty_repr(parsing_table.to_pretty_table()))
        root = ParseTree(self.grammar.start, [])
        stack, token_index = [
            (EOF, root),
            (self.grammar.start, root),
        ], 0

        while stack:
            symbol, root = stack.pop()
            token = tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
                    root.expansion.append(token)
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.name} but got {token}")
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
                        f'expecting one of ({", ".join(terminal.name for terminal in self.grammar.gen_first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert token_index >= len(tokens)
        return root


class EarleyParser(Parser):
    def parse(self, tokens: list[Token]) -> Iterator[ParseTree]:
        """Parse a list of tokens into a parse tree"""

        earley_sets = [
            LRState[EarleyItem](*earley_set.yield_finished(), cls=EarleyItem)
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
                and earley_item.name == self.grammar.start
            ):
                for tree in build_tree(earley_item, 0, n_tokens):
                    yield tree


class LR0Parser(Parser):
    def get_parsing_table(self):
        return LR0ParsingTable(self.grammar)

    def parse(self, tokens: list[Token]) -> Iterator[ParseTree] | ParseTree:
        parsing_table = self.get_parsing_table()
        # root = ParseTree(self.grammar.start_symbol, [])
        stack, token_index = [parsing_table.states[0]], 0
        tree: list[ParseTree | Token] = []

        while stack:
            current_state = stack[-1]
            current_token = tokens[token_index]
            match parsing_table.get((current_state, current_token.id)):
                # Advance input one token; push state n on stack.
                # TODO: assert that current_state corresponds to the current_token
                case Shift(current_state):
                    stack.append(current_state)
                    tree.append(current_token)
                    token_index += current_token.id != EOF.name
                case Reduce(lhs, len_rhs):
                    stack = stack[:-len_rhs]
                    match parsing_table[(stack[-1], lhs.name)]:
                        case Goto(current_state):
                            stack.append(current_state)
                            tree_top = tree[-len_rhs:]
                            tree = tree[:-len_rhs] + [ParseTree(lhs, tree_top)]
                        case _:
                            raise SyntaxError(
                                f"Unexpected {current_token.id} at {current_token.loc}"
                            )
                case Accept():
                    return one(tree)
                case _:
                    raise SyntaxError(
                        f"Unexpected {current_token.id} at {current_token.loc}"
                    )
        raise SyntaxError(
            f"Syntax error at {tokens[token_index] if token_index < len(tokens) else EOF}"
        )


class SLRParser(LR0Parser):
    def get_parsing_table(self):
        return SLRParsingTable(self.grammar)


class LR1Parser(LR0Parser):
    def get_parsing_table(self):
        return LR1ParsingTable(self.grammar)


class LALR1Parser(LR0Parser):
    def get_parsing_table(self):
        return LALR1ParsingTable(self.grammar)


if __name__ == "__main__":
    from utils.parse_grammar import parse_grammar

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
    #
    # table = {
    #     "+": "+",
    #     "-": "-",
    #     "*": "*",
    #     "/": "/",
    #     "(": "(",
    #     ")": ")",
    #     "^": "^",
    # }
    # table = {
    #     "(": "(",
    #     ")": ")",
    #     "-": "-",
    #     "/": "/",
    #     "or_literal": "|",
    #     "?:": "capture",
    #     "$": "end",
    #     "^": "start",
    #     ".": "any",
    #     "\\d": "digit",
    #     "\\D": "not_digit",
    #     "\\s": "space",
    #     "\\S": "not_space",
    #     "\\w": "word",
    #     "\\W": "not_word",
    # }
    #
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
    # cfg = parse_grammar(g, table)
    # print_rich(pretty_repr(cfg))
    #
    # # tks = Tokenizer("a + a − a", table).get_tokens_no_whitespace()
    # #
    # tks = Tokenizer("^a+(?:ab)c{1,3}$", table).get_tokens_no_whitespace()
    # # tks = Tokenizer("(1 + 1)", table).get_tokens_no_whitespace()
    # print_rich(pretty_repr(tks))
    #
    # earley_parser = EarleyParser(cfg)
    # trees = list(earley_parser.parse(tks))
    # print_rich(pretty_repr([t.collapse() for t in trees]))
    # table = {
    #     '+': '+',
    #     '*': '*',
    #     '(': '(',
    #     ')': ')',
    # }
    #
    # g = """
    # <E'>
    # <E'> -> <E>
    # <E> -> <E>+<T>
    # <E> -> <T>
    # <T> -> <T>*<F>
    # <T> -> <F>
    # <F> -> (<E>)
    # <F> -> integer
    #
    # """
    #
    # cfg = parse_grammar(g, table)
    # print_rich(pretty_repr(cfg))
    #
    # # p = SLRParsingTable(cfg)
    # # p.draw_with_graphviz()
    # # print_rich(p.to_pretty_table())
    #
    # tks = Tokenizer("(1+2)", table).get_tokens_no_whitespace()
    # print_rich(pretty_repr(tks))
    # print_rich(pretty_repr(LR1Parser(cfg).parse(tks).collapse()))

    table = {
        "(": "(",
        ")": ")",
        "+": "+",
        "*": "*",
    }

    g = """
    <E>
    <E> -> <T> <E'>
    <E'> -> + <T> <E'> | <>
    <T> -> <F> <T'>
    <T'> -> * <F> <T'> | <>
    <F> -> ( <E> ) | integer
    
    """

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))

    print_rich(pretty_repr(cfg.gen_follow()))

    print_rich(pretty_repr(cfg.gen_nullable()))
    print_rich(pretty_repr(cfg.gen_follow()))
    tks = Tokenizer("(1+2)", table).get_tokens_no_whitespace()
    p = LL1Parser(cfg)
    print_rich(pretty_repr(p.parse(tks).collapse()))
