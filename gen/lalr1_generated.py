"""
    ID: 5f21cf4f65a321b75a0c9d6bf9daee4926cbece0f590311ac32070ab74769de1
"""
from more_itertools import one
from rich.traceback import install

from grammar import EOF, Terminal
from parsers.parser import ParseTree
from tokenizer import Tokenizer

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


tokenizer_table: dict[str, str] = {"(": "(", ")": ")", "+": "+", "*": "*"}  # type: ignore

parsing_table: dict[tuple[int, str], Action] = {
    (1, "("): 15,
    (1, "F"): 10,
    (1, "T"): 6,
    (1, "integer"): 39,
    (1, "E"): 4,
    (2, "eof"): -1,
    (3, "+"): 23,
    (3, ")"): ("N_1", 0),
    (3, "R_0"): 20,
    (3, "E0"): 8,
    (3, "N_1"): 18,
    (3, "eof"): ("N_1", 0),
    (4, ")"): ("E", 2),
    (4, "eof"): ("E", 2),
    (5, "+"): ("N_3", 0),
    (5, ")"): ("N_3", 0),
    (5, "N_3"): 28,
    (5, "T0"): 12,
    (5, "*"): 33,
    (5, "R_2"): 30,
    (5, "eof"): ("N_3", 0),
    (6, "+"): ("T", 2),
    (6, ")"): ("T", 2),
    (6, "eof"): ("T", 2),
    (7, "("): 15,
    (7, "F"): 10,
    (7, "T"): 6,
    (7, "integer"): 39,
    (7, "E"): 16,
    (8, ")"): 41,
    (9, ")"): ("E0", 1),
    (9, "eof"): ("E0", 1),
    (10, ")"): ("N_1", 1),
    (10, "eof"): ("N_1", 1),
    (11, "("): 15,
    (11, "F"): 10,
    (11, "T"): 24,
    (11, "integer"): 39,
    (12, "+"): 23,
    (12, ")"): ("N_1", 0),
    (12, "R_0"): 20,
    (12, "E0"): 26,
    (12, "N_1"): 18,
    (12, "eof"): ("N_1", 0),
    (13, ")"): ("R_0", 3),
    (13, "eof"): ("R_0", 3),
    (14, "+"): ("T0", 1),
    (14, ")"): ("T0", 1),
    (14, "eof"): ("T0", 1),
    (15, "+"): ("N_3", 1),
    (15, ")"): ("N_3", 1),
    (15, "eof"): ("N_3", 1),
    (16, "("): 15,
    (16, "F"): 34,
    (16, "integer"): 39,
    (17, "+"): ("N_3", 0),
    (17, ")"): ("N_3", 0),
    (17, "N_3"): 28,
    (17, "T0"): 36,
    (17, "*"): 33,
    (17, "R_2"): 30,
    (17, "eof"): ("N_3", 0),
    (18, "+"): ("R_2", 3),
    (18, ")"): ("R_2", 3),
    (18, "eof"): ("R_2", 3),
    (19, "+"): ("F", 1),
    (19, ")"): ("F", 1),
    (19, "*"): ("F", 1),
    (19, "eof"): ("F", 1),
    (20, "+"): ("F", 3),
    (20, ")"): ("F", 3),
    (20, "*"): ("F", 3),
    (20, "eof"): ("F", 3),
}  # type: ignore

states: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # type: ignore

expected_tokens: dict[int, list[str]] = {
    1: ["(", "integer"],
    2: ["eof"],
    3: ["+", ")", "eof"],
    4: [")", "eof"],
    5: ["+", ")", "*", "eof"],
    6: ["+", ")", "eof"],
    7: ["(", "integer"],
    8: [")"],
    9: [")", "eof"],
    10: [")", "eof"],
    11: ["(", "integer"],
    12: ["+", ")", "eof"],
    13: [")", "eof"],
    14: ["+", ")", "eof"],
    15: ["+", ")", "eof"],
    16: ["(", "integer"],
    17: ["+", ")", "*", "eof"],
    18: ["+", ")", "eof"],
    19: ["+", ")", "*", "eof"],
    20: ["+", ")", "*", "eof"],
}  # type: ignore


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
