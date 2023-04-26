"""
    ID: %id%
"""
from more_itertools import one
from rich.traceback import install

from grammar import EOF, Terminal
from parsers.parser import ParseTree

Shift = Goto = Accept = int
Reduce = tuple[str, int]
Action = int | Reduce


install(show_locals=True)


def is_accept(act: int) -> bool:
    return act == -1


def is_goto(act: int) -> bool:
    return act & 0b1 == 0b0


def is_shift(act: int) -> bool:
    return act & 0b1 == 0b1


parsing_table: dict[tuple[int, str], Action] = "%parsing_table%"  # type: ignore

states: list[int] = "%states%"  # type: ignore

expected_tokens: dict[int, list[str]] = "%expected_tokens%"  # type: ignore

tokenizer: Tokenizer = "%tokenizer_table%"  # type: ignore


def parse(input_str: str) -> ParseTree:
    tokens = tokenizer.get_tokens_no_whitespace(input_str)
    stack, token_index = [states[0]], 0
    tree: list[ParseTree | Terminal] = []

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
            case (str(lhs), int(len_rhs)):
                stack = stack[: -len_rhs or None]
                act = parsing_table[(stack[-1], lhs)]
                assert isinstance(act, int) and is_goto(act)
                stack.append(act >> 0b1)
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
