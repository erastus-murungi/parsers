"""
    ID: 904df7ff5fb8d0ca1c2b0eedc348ad685cd14b99de062d77cceb85b106d32dbc
"""
from more_itertools import one
from rich.traceback import install

from grammar import EOF, Terminal
from parsers.parser import ParseTree
from tokenizer import Tokenizer

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


tokenizer_table: dict[str, str] = {"(": "(", ")": ")", "+": "+", "*": "*"}

parsing_table: dict[tuple[int, str], Action] = {
    (1, "T"): 6,
    (1, "F"): 10,
    (1, "("): 15,
    (1, "integer"): 39,
    (1, "E"): 4,
    (2, "eof"): -1,
    (3, "N_1"): 18,
    (3, "eof"): ("N_1", 0),
    (3, ")"): ("N_1", 0),
    (3, "E0"): 8,
    (3, "+"): 23,
    (3, "R_0"): 20,
    (4, "eof"): ("E", 2),
    (4, ")"): ("E", 2),
    (5, "R_2"): 30,
    (5, "*"): 33,
    (5, "eof"): ("N_3", 0),
    (5, ")"): ("N_3", 0),
    (5, "+"): ("N_3", 0),
    (5, "T0"): 12,
    (5, "N_3"): 28,
    (6, "eof"): ("T", 2),
    (6, ")"): ("T", 2),
    (6, "+"): ("T", 2),
    (7, "T"): 6,
    (7, "F"): 10,
    (7, "("): 15,
    (7, "integer"): 39,
    (7, "E"): 16,
    (8, ")"): 41,
    (9, "eof"): ("E0", 1),
    (9, ")"): ("E0", 1),
    (10, "eof"): ("N_1", 1),
    (10, ")"): ("N_1", 1),
    (11, "T"): 24,
    (11, "F"): 10,
    (11, "("): 15,
    (11, "integer"): 39,
    (12, "N_1"): 18,
    (12, "eof"): ("N_1", 0),
    (12, ")"): ("N_1", 0),
    (12, "E0"): 26,
    (12, "+"): 23,
    (12, "R_0"): 20,
    (13, "eof"): ("R_0", 3),
    (13, ")"): ("R_0", 3),
    (14, "eof"): ("T0", 1),
    (14, ")"): ("T0", 1),
    (14, "+"): ("T0", 1),
    (15, "eof"): ("N_3", 1),
    (15, ")"): ("N_3", 1),
    (15, "+"): ("N_3", 1),
    (16, "F"): 34,
    (16, "("): 15,
    (16, "integer"): 39,
    (17, "R_2"): 30,
    (17, "*"): 33,
    (17, "eof"): ("N_3", 0),
    (17, ")"): ("N_3", 0),
    (17, "+"): ("N_3", 0),
    (17, "T0"): 36,
    (17, "N_3"): 28,
    (18, "eof"): ("R_2", 3),
    (18, ")"): ("R_2", 3),
    (18, "+"): ("R_2", 3),
    (19, "*"): ("F", 1),
    (19, "eof"): ("F", 1),
    (19, ")"): ("F", 1),
    (19, "+"): ("F", 1),
    (20, "*"): ("F", 3),
    (20, "eof"): ("F", 3),
    (20, ")"): ("F", 3),
    (20, "+"): ("F", 3),
}

states: list[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
]

expected_tokens: dict[int, list[str]] = {
    1: ["(", "integer"],
    2: ["eof"],
    3: ["eof", ")", "+"],
    4: ["eof", ")"],
    5: ["*", "eof", ")", "+"],
    6: ["eof", ")", "+"],
    7: ["(", "integer"],
    8: [")"],
    9: ["eof", ")"],
    10: ["eof", ")"],
    11: ["(", "integer"],
    12: ["eof", ")", "+"],
    13: ["eof", ")"],
    14: ["eof", ")", "+"],
    15: ["eof", ")", "+"],
    16: ["(", "integer"],
    17: ["*", "eof", ")", "+"],
    18: ["eof", ")", "+"],
    19: ["*", "eof", ")", "+"],
    20: ["*", "eof", ")", "+"],
}


def parse(input_str: str) -> ParseTree:
    tokens = Tokenizer(input_str, tokenizer_table).get_tokens_no_whitespace()
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
