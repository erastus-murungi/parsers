"""
    ID: f97d30f9b3c26910757f46c430b68f3e30f822eb6a66992f893e39e2f4715e2b
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


tokenizer_table: dict[str, str] = {
    "int": "int",
    "char": "char",
    "float": "float",
    "if": "if",
    "else": "else",
    "while": "while",
    "return": "return",
    "break": "break",
    "continue": "continue",
    "(": ")",
    ")": ")",
    "{": "{",
    "}": "}",
    "[": "[",
    "]": "]",
    ";": ";",
    "=": "=",
    ",": ",",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
    "&&": "&&",
    "!": "!",
    "++": "++",
    "--": "--",
    "*=": "*=",
    "/=": "/=",
    "%=": "%=",
    "+=": "+=",
    "-=": "-=",
    "<<=": "<<=",
    "double_or": "||",
}

parsing_table: dict[tuple[int, str], Action] = {
    (1, "import"): ("N_0", 0),
    (1, "N_0"): 4,
    (1, "eof"): ("N_0", 0),
    (2, "import"): 15,
    (2, "eof"): -1,
    (2, "import_decl"): 6,
    (3, "import"): ("N_0", 2),
    (3, "eof"): ("N_0", 2),
    (4, ";"): ("N_4", 0),
    (4, ","): ("N_4", 0),
    (4, "N_4"): 10,
    (5, ";"): 17,
    (5, ","): 19,
    (5, "R_3"): 12,
    (6, ";"): ("N_4", 2),
    (6, ","): ("N_4", 2),
    (7, "word"): 9,
    (8, "import"): ("import_decl", 4),
    (8, "eof"): ("import_decl", 4),
    (9, "word"): 21,
    (10, ";"): ("R_3", 2),
    (10, ","): ("R_3", 2),
}

states: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

expected_tokens: dict[int, list[str]] = {
    1: ["import", "eof"],
    2: ["import", "eof"],
    3: ["import", "eof"],
    4: [";", ","],
    5: [";", ","],
    6: [";", ","],
    7: ["word"],
    8: ["import", "eof"],
    9: ["word"],
    10: [";", ","],
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
