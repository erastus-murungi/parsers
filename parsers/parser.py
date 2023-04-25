from abc import ABC, abstractmethod
from itertools import product
from typing import Iterator, NamedTuple, Required, Sequence, TypedDict, Union, cast

from more_itertools import one
from rich.traceback import install

from earley import EarleyItem, gen_earley_sets
from grammar import EMPTY, EOF, Grammar, NonTerminal, Symbol, Terminal
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
from tokenizer.tokenizer import Tokenizer

install(show_locals=False)


class AST(TypedDict):
    id: Required[str]
    expansion: Required[list[Union["AST", Terminal]]]


class ParseTree(NamedTuple):
    id: str
    expansion: list[Union["ParseTree", Terminal]]

    def collapse(self) -> AST:
        expansion: list[AST | Terminal] = []
        for child in self.expansion:
            if isinstance(child, Terminal):
                expansion.append(child)
            else:
                child_collapse = child.collapse()
                assert "id" in child_collapse
                if child_collapse:
                    if len(child_collapse["expansion"]) <= 1:
                        expansion.extend(child_collapse["expansion"])
                    else:
                        expansion.append(child_collapse)
        return {"id": self.id, "expansion": expansion}


class Parser(ABC):
    def __init__(self, grammar: Grammar, source: str, table: dict):
        self.grammar = grammar
        self.source = source
        self.tokens = Tokenizer(source, table).get_tokens_no_whitespace()

    @abstractmethod
    def parse(
        self,
    ) -> Iterator[ParseTree] | ParseTree:
        """Parse a list of tokens into a parse tree"""
        ...


class LL1Parser(Parser):
    def parse(self) -> Iterator[ParseTree] | ParseTree:
        parsing_table = LL1ParsingTable(self.grammar)
        root = ParseTree(str(self.grammar.start), [])
        stack, token_index = [
            (EOF, root),
            (self.grammar.start, root),
        ], 0

        while stack:
            symbol, root = stack.pop()
            token = self.tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
                    root.expansion.append(token)
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.name} but got {token}")
            else:
                non_terminal = cast(NonTerminal, symbol)
                if (rule := parsing_table.get((non_terminal, token.id))) is not None:
                    nodes = [ParseTree(str(sym), []) for sym in rule]
                    root.expansion.extend(nodes)
                    stack.extend(reversed(list(zip(rule, nodes))))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.name for terminal in self.grammar.gen_first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert token_index >= len(self.tokens)
        return root


class EarleyParser(Parser):
    def parse(self) -> Iterator[ParseTree]:
        """Parse a list of tokens into a parse tree"""

        earley_sets = [
            LRState[EarleyItem](*earley_set.yield_finished(), cls=EarleyItem)
            for earley_set in gen_earley_sets(self.grammar, self.tokens, self.source)
        ]

        # reverse the earley sets
        parse_forest: list[list[EarleyItem]] = [[] for _ in range(len(self.tokens))]
        for end_index, earley_set in enumerate(earley_sets):
            for earley_item in earley_set:
                name, dot, start, rule = earley_item
                parse_forest[start].append(EarleyItem(name, dot, end_index, rule))

        def yield_all_paths(
            path: list[EarleyItem | Terminal],
            start_index: int,
            left,
            path_end_index: int,
        ) -> Iterator[list[EarleyItem | Terminal]]:
            # if we are at a leaf
            if not left:
                if path_end_index == start_index:
                    yield path
            else:
                current_symbol, *left = left
                if isinstance(current_symbol, Terminal):
                    current_token = self.tokens[start_index]
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
                children_possibilities: list[Sequence[Terminal | ParseTree]] = []
                for item in path:
                    if isinstance(item, Terminal):
                        children_possibilities.append([item])
                        item_start_index += 1
                    else:
                        child_tree = [
                            t
                            for t in build_tree(
                                item, item_start_index, item.explicit_index
                            )
                            if t.expansion
                        ]
                        if child_tree:
                            children_possibilities.append(child_tree)
                            item_start_index = item.explicit_index
                for children in product(*children_possibilities):
                    yield ParseTree(str(path_root.name), list(children))

        n_tokens = len(self.tokens) - 1  # ignore EOF
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

    def parse(self) -> Iterator[ParseTree] | ParseTree:
        parsing_table = self.get_parsing_table()
        # root = ParseTree(self.grammar.start_symbol, [])
        stack, token_index = [parsing_table.states[0]], 0
        tree: list[ParseTree | Terminal] = []

        while stack:
            current_state = stack[-1]
            current_token = self.tokens[token_index]
            match parsing_table.get((current_state, current_token.id)):
                # Advance input one token; push state n on stack.
                # TODO: assert that current_state corresponds to the current_token
                case Shift(current_state):
                    stack.append(current_state)
                    tree.append(current_token)
                    token_index += current_token.id != EOF.name
                case Reduce(lhs, len_rhs):
                    stack = stack[: -len_rhs or None]
                    match parsing_table[(stack[-1], lhs.name)]:
                        case Goto(current_state):
                            stack.append(current_state)
                            tree_top = tree[-len_rhs:]
                            tree = tree[:-len_rhs] + [ParseTree(str(lhs), tree_top)]
                        case _:
                            raise SyntaxError(
                                f"Unexpected {current_token.id} at {current_token.loc}"
                            )
                case Accept():
                    root = one(tree)
                    assert isinstance(root, ParseTree)
                    return root
                case _:
                    raise SyntaxError(
                        f"Unexpected {current_token.id} at {current_token.loc}"
                    )
        raise SyntaxError(
            f"Syntax error at {self.tokens[token_index] if token_index < len(self.tokens) else EOF}"
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
