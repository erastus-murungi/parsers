"""
    ID: 13f9bf1b2f70f541722783cdf91a6b63d643983bbfde2ffa266286170048395a
"""

import re

from more_itertools import one

from grammar import EOF, Terminal, Tokenizer
from parsers.parser import ParseTree

patterns: dict[str, re.Pattern] = {
    "+": re.compile("\\+", re.DOTALL),
    "*": re.compile("\\*", re.DOTALL),
    "(": re.compile("\\(", re.DOTALL),
    ")": re.compile("\\)", re.DOTALL),
    "number": re.compile(
        "((([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))",
        re.DOTALL,
    ),
}  # type: ignore

reserved: frozenset[str] = frozenset()  # type: ignore


Shift = Goto = Accept = int
Reduce = tuple[str, int]
Action = int | Reduce


def is_accept(act: int) -> bool:
    return act == -1


def is_goto(act: int) -> bool:
    return act & 0b1 == 0b0


def is_shift(act: int) -> bool:
    return act & 0b1 == 0b1


parsing_table: dict[tuple[int, str], Action] = {
    (1, "F"): 10,
    (1, "T"): 6,
    (1, "("): 15,
    (1, "E"): 4,
    (1, "number"): 31,
    (2, "eof"): -1,
    (3, "eof"): ("E0", 0),
    (3, ")"): ("E0", 0),
    (3, "+"): 19,
    (3, "E0"): 8,
    (4, "eof"): ("E", 2),
    (4, ")"): ("E", 2),
    (5, "eof"): ("T0", 0),
    (5, "T0"): 12,
    (5, "*"): 25,
    (5, ")"): ("T0", 0),
    (5, "+"): ("T0", 0),
    (6, "eof"): ("T", 2),
    (6, ")"): ("T", 2),
    (6, "+"): ("T", 2),
    (7, "F"): 10,
    (7, "T"): 6,
    (7, "("): 15,
    (7, "E"): 16,
    (7, "number"): 31,
    (8, ")"): 33,
    (9, "F"): 10,
    (9, "T"): 20,
    (9, "("): 15,
    (9, "number"): 31,
    (10, "eof"): ("E0", 0),
    (10, ")"): ("E0", 0),
    (10, "+"): 19,
    (10, "E0"): 22,
    (11, "eof"): ("E0", 3),
    (11, ")"): ("E0", 3),
    (12, "F"): 26,
    (12, "("): 15,
    (12, "number"): 31,
    (13, "eof"): ("T0", 0),
    (13, "T0"): 28,
    (13, "*"): 25,
    (13, ")"): ("T0", 0),
    (13, "+"): ("T0", 0),
    (14, "eof"): ("T0", 3),
    (14, ")"): ("T0", 3),
    (14, "+"): ("T0", 3),
    (15, "eof"): ("F", 1),
    (15, "*"): ("F", 1),
    (15, ")"): ("F", 1),
    (15, "+"): ("F", 1),
    (16, "eof"): ("F", 3),
    (16, "*"): ("F", 3),
    (16, ")"): ("F", 3),
    (16, "+"): ("F", 3),
}  # type: ignore

states: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # type: ignore

expected_tokens: dict[int, list[str]] = {
    1: ["(", "number"],
    2: ["eof"],
    3: ["eof", ")", "+"],
    4: ["eof", ")"],
    5: ["eof", "*", ")", "+"],
    6: ["eof", ")", "+"],
    7: ["(", "number"],
    8: [")"],
    9: ["(", "number"],
    10: ["eof", ")", "+"],
    11: ["eof", ")"],
    12: ["(", "number"],
    13: ["eof", "*", ")", "+"],
    14: ["eof", ")", "+"],
    15: ["eof", "*", ")", "+"],
    16: ["eof", "*", ")", "+"],
}  # type: ignore

tokenizer = Tokenizer(patterns, reserved, filename="(void)")  # type: ignore


def parse(input_str: str) -> ParseTree:
    tokens = tokenizer.get_tokens_no_whitespace(input_str)
    stack, token_index = [states[0]], 0
    tree: list[ParseTree | Terminal] = []

    while stack:
        state = stack[-1]
        token = tokens[token_index]
        match parsing_table.get((state, token.name)):
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
                    f"of type {token.name} at {token.loc}\n"
                    f"Expected one of {expected_tokens.get(state, [])}"
                )
    raise SyntaxError(
        f"Syntax error at {tokens[token_index] if token_index < len(tokens) else EOF}"
    )
