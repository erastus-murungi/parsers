"""
    ID: b71444ce0430a010f78e2d19bfc3be5d5bcb65a170daf1dda07910556a0bdb97
"""
from more_itertools import one

from grammar import EOF
from parsers.parser import ParseTree
from utils import Token, Tokenizer
from rich.traceback import install

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
    (1, "F"): 10,
    (1, "E"): 4,
    (1, "integer"): 31,
    (1, "T"): 6,
    (1, "("): 15,
    (2, "eof"): -1,
    (3, "E'"): 8,
    (3, "+"): 19,
    (3, ")"): ("E'", 0),
    (3, "eof"): ("E'", 0),
    (4, ")"): ("E", 2),
    (4, "eof"): ("E", 2),
    (5, "*"): 25,
    (5, "+"): ("T'", 0),
    (5, ")"): ("T'", 0),
    (5, "eof"): ("T'", 0),
    (5, "T'"): 12,
    (6, "+"): ("T", 2),
    (6, ")"): ("T", 2),
    (6, "eof"): ("T", 2),
    (7, "F"): 10,
    (7, "E"): 16,
    (7, "integer"): 31,
    (7, "T"): 6,
    (7, "("): 15,
    (8, ")"): 33,
    (9, "F"): 10,
    (9, "integer"): 31,
    (9, "T"): 20,
    (9, "("): 15,
    (10, "E'"): 22,
    (10, "+"): 19,
    (10, ")"): ("E'", 0),
    (10, "eof"): ("E'", 0),
    (11, ")"): ("E'", 3),
    (11, "eof"): ("E'", 3),
    (12, "F"): 26,
    (12, "integer"): 31,
    (12, "("): 15,
    (13, "*"): 25,
    (13, "+"): ("T'", 0),
    (13, ")"): ("T'", 0),
    (13, "eof"): ("T'", 0),
    (13, "T'"): 28,
    (14, "+"): ("T'", 3),
    (14, ")"): ("T'", 3),
    (14, "eof"): ("T'", 3),
    (15, "*"): ("F", 1),
    (15, "+"): ("F", 1),
    (15, ")"): ("F", 1),
    (15, "eof"): ("F", 1),
    (16, "*"): ("F", 3),
    (16, "+"): ("F", 3),
    (16, ")"): ("F", 3),
    (16, "eof"): ("F", 3),
}

states: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

expected_tokens: dict[int, list[str]] = {
    1: ["integer", "("],
    2: ["eof"],
    3: ["+", ")", "eof"],
    4: [")", "eof"],
    5: ["*", "+", ")", "eof"],
    6: ["+", ")", "eof"],
    7: ["integer", "("],
    8: [")"],
    9: ["integer", "("],
    10: ["+", ")", "eof"],
    11: [")", "eof"],
    12: ["integer", "("],
    13: ["*", "+", ")", "eof"],
    14: ["+", ")", "eof"],
    15: ["*", "+", ")", "eof"],
    16: ["*", "+", ")", "eof"],
}


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


if __name__ == '__main__':
    print(parse("(1+2)*3"))
