"""
    ID: 49f150b25a1f32bb54ebe230d5c29b45714cbbaee2d23b0dfa36e9c61d672a64
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


tokenizer_table: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "(": "(",
    ")": ")",
    "^": "^",
}

parsing_table: dict[tuple[int, str], Action] = {
    (1, "expression"): 6,
    (1, "digit"): 32,
    (1, "integer"): 39,
    (1, "float"): 37,
    (1, "("): 27,
    (1, "power"): 20,
    (1, "term"): 8,
    (1, "number"): 30,
    (1, "program"): 4,
    (1, "factor"): 14,
    (2, "eof"): -1,
    (3, "eof"): ("program", 1),
    (4, ")"): ("expression", 1),
    (4, "+"): 43,
    (4, "add_op"): 10,
    (4, "eof"): ("expression", 1),
    (4, "-"): 41,
    (5, "expression"): 12,
    (5, "digit"): 32,
    (5, "integer"): 39,
    (5, "float"): 37,
    (5, "("): 27,
    (5, "power"): 20,
    (5, "term"): 8,
    (5, "number"): 30,
    (5, "factor"): 14,
    (6, ")"): ("expression", 3),
    (6, "eof"): ("expression", 3),
    (7, "mult_op"): 16,
    (7, "/"): 45,
    (7, ")"): ("term", 1),
    (7, "+"): ("term", 1),
    (7, "*"): 47,
    (7, "eof"): ("term", 1),
    (7, "-"): ("term", 1),
    (8, "digit"): 32,
    (8, "integer"): 39,
    (8, "float"): 37,
    (8, "("): 27,
    (8, "power"): 20,
    (8, "term"): 18,
    (8, "number"): 30,
    (8, "factor"): 14,
    (9, ")"): ("term", 3),
    (9, "+"): ("term", 3),
    (9, "eof"): ("term", 3),
    (9, "-"): ("term", 3),
    (10, "/"): ("factor", 1),
    (10, ")"): ("factor", 1),
    (10, "+"): ("factor", 1),
    (10, "*"): ("factor", 1),
    (10, "^"): 23,
    (10, "eof"): ("factor", 1),
    (10, "-"): ("factor", 1),
    (11, "digit"): 32,
    (11, "integer"): 39,
    (11, "float"): 37,
    (11, "("): 27,
    (11, "power"): 20,
    (11, "number"): 30,
    (11, "factor"): 24,
    (12, "/"): ("factor", 3),
    (12, ")"): ("factor", 3),
    (12, "+"): ("factor", 3),
    (12, "*"): ("factor", 3),
    (12, "eof"): ("factor", 3),
    (12, "-"): ("factor", 3),
    (13, "expression"): 28,
    (13, "digit"): 32,
    (13, "integer"): 39,
    (13, "float"): 37,
    (13, "("): 27,
    (13, "power"): 20,
    (13, "term"): 8,
    (13, "number"): 30,
    (13, "factor"): 14,
    (14, ")"): 49,
    (15, "/"): ("power", 1),
    (15, ")"): ("power", 1),
    (15, "+"): ("power", 1),
    (15, "*"): ("power", 1),
    (15, "^"): ("power", 1),
    (15, "eof"): ("power", 1),
    (15, "-"): ("power", 1),
    (16, "/"): ("number", 1),
    (16, ")"): ("number", 1),
    (16, "+"): ("number", 1),
    (16, "*"): ("number", 1),
    (16, "^"): ("number", 1),
    (16, "eof"): ("number", 1),
    (16, "digit"): 32,
    (16, "integer"): 39,
    (16, "float"): 37,
    (16, "-"): ("number", 1),
    (16, "number"): 34,
    (17, "/"): ("number", 2),
    (17, ")"): ("number", 2),
    (17, "+"): ("number", 2),
    (17, "*"): ("number", 2),
    (17, "^"): ("number", 2),
    (17, "eof"): ("number", 2),
    (17, "-"): ("number", 2),
    (18, "/"): ("digit", 1),
    (18, ")"): ("digit", 1),
    (18, "+"): ("digit", 1),
    (18, "*"): ("digit", 1),
    (18, "^"): ("digit", 1),
    (18, "eof"): ("digit", 1),
    (18, "integer"): ("digit", 1),
    (18, "float"): ("digit", 1),
    (18, "-"): ("digit", 1),
    (19, "/"): ("digit", 1),
    (19, ")"): ("digit", 1),
    (19, "+"): ("digit", 1),
    (19, "*"): ("digit", 1),
    (19, "^"): ("digit", 1),
    (19, "eof"): ("digit", 1),
    (19, "integer"): ("digit", 1),
    (19, "float"): ("digit", 1),
    (19, "-"): ("digit", 1),
    (20, "integer"): ("add_op", 1),
    (20, "float"): ("add_op", 1),
    (20, "("): ("add_op", 1),
    (21, "integer"): ("add_op", 1),
    (21, "float"): ("add_op", 1),
    (21, "("): ("add_op", 1),
    (22, "integer"): ("mult_op", 1),
    (22, "float"): ("mult_op", 1),
    (22, "("): ("mult_op", 1),
    (23, "integer"): ("mult_op", 1),
    (23, "float"): ("mult_op", 1),
    (23, "("): ("mult_op", 1),
    (24, "/"): ("power", 3),
    (24, ")"): ("power", 3),
    (24, "+"): ("power", 3),
    (24, "*"): ("power", 3),
    (24, "^"): ("power", 3),
    (24, "eof"): ("power", 3),
    (24, "-"): ("power", 3),
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
    21,
    22,
    23,
    24,
]

expected_tokens: dict[int, list[str]] = {
    1: ["integer", "float", "("],
    2: ["eof"],
    3: ["eof"],
    4: [")", "+", "eof", "-"],
    5: ["integer", "float", "("],
    6: [")", "eof"],
    7: ["/", ")", "+", "*", "eof", "-"],
    8: ["integer", "float", "("],
    9: [")", "+", "eof", "-"],
    10: ["/", ")", "+", "*", "^", "eof", "-"],
    11: ["integer", "float", "("],
    12: ["/", ")", "+", "*", "eof", "-"],
    13: ["integer", "float", "("],
    14: [")"],
    15: ["/", ")", "+", "*", "^", "eof", "-"],
    16: ["/", ")", "+", "*", "^", "eof", "integer", "float", "-"],
    17: ["/", ")", "+", "*", "^", "eof", "-"],
    18: ["/", ")", "+", "*", "^", "eof", "integer", "float", "-"],
    19: ["/", ")", "+", "*", "^", "eof", "integer", "float", "-"],
    20: ["integer", "float", "("],
    21: ["integer", "float", "("],
    22: ["integer", "float", "("],
    23: ["integer", "float", "("],
    24: ["/", ")", "+", "*", "^", "eof", "-"],
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


if __name__ == "__main__":
    print(parse("1 + 2 * 3 / ( 4 ^ 4 )"))
