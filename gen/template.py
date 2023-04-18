"""
    ID: %id%
"""
from more_itertools import one
from rich.traceback import install

from grammar import EOF
from parsers.parser import ParseTree
from utils import Token, Tokenizer

Shift = Goto = Accept = int
Reduce = tuple[int, int]
Action = Shift | Goto | Accept | Reduce


install(show_locals=True)


def is_accept(act: Action) -> bool:
    return act == -1


def is_goto(act: Action) -> bool:
    return act & 0b1 == 0b0


def is_shift(act: Action) -> bool:
    return act & 0b1 == 0b1


tokenizer_table: dict[str, str] = "%tokenizer_table%"

parsing_table: dict[tuple[int, str], Action] = "%parsing_table%"

states: list[int] = "%states%"

expected_tokens: dict[int, list[str]] = "%expected_tokens%"


def parse(input_str: str) -> ParseTree:
    tokens = Tokenizer(input_str, tokenizer_table).get_tokens_no_whitespace()
    stack, token_index = [states[0]], 0
    tree: list[ParseTree | Token] = []

    while stack:
        state = stack[-1]
        token = tokens[token_index]
        match parsing_table.get((state, token.id)):
            case int(action) if is_accept(action):
                root = one(tree)
                assert isinstance(root, ParseTree)
                return root
            case int(action) if is_shift(action):
                stack.append(action >> 0b1)
                tree.append(token)
                token_index += 1
            case (lhs, len_rhs):
                stack = stack[: -len_rhs or None]
                action = parsing_table[(stack[-1], lhs)]
                assert is_goto(action)
                stack.append(action >> 0b1)
                tree = tree[:-len_rhs] + [ParseTree(lhs, tree[-len_rhs:])]
            case _:
                raise SyntaxError(
                    f"Encountered unexpected token `{token.lexeme}` "
                    f"of type {token.id} at {token.loc}\n"
                    f"Expected one of {expected_tokens.get(state, [])}"
                )
    raise SyntaxError(
        f"Syntax error at {tokens[token_index] if token_index < len(tokens) else EOF}"
    )
