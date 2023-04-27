"""
    ID: d4a68c33bc74d9849a99fb359e93922ed8182ba422fcd70a0237e6d06c2b261f
"""

import re
from typing import Iterator

from more_itertools import one

from grammar import EOF, Loc, Terminal
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
}

reserved: frozenset[str] = frozenset()


class Tokenizer:
    def __init__(
        self,
        filename: str = "(void)",
    ):
        self._filename = filename
        self._code = ""
        self._linenum = 0
        self._column = 0
        self._code_offset = 0
        self.patterns = patterns
        self.reserved = reserved

    def get_filename(self):
        return self._filename

    def _reset(self, code: str, filename: str = "%filename"):
        self._filename = filename
        self._code = code + "\n" if code and code[-1] != "\n" else code
        self._linenum = 0
        self._column = 0
        self._code_offset = 0

    def _to_next_char(self):
        self._code_offset += 1
        self._column += 1

    def _skip_n_chars(self, n):
        self._code_offset += n
        self._column += n

    def _current_char(self):
        return self._code[self._code_offset]

    def _remaining_code(self):
        return self._code[self._code_offset :]

    def _tokenize(self) -> Iterator[Terminal]:
        while self._code_offset < len(self._code):
            token_location = Loc(
                self._filename, self._linenum, self._column, self._code_offset
            )
            # greedy attempt
            matches: list[tuple[str, str]] = []
            for identifier, pattern in self.patterns.items():
                matching = pattern.match(self._remaining_code())
                if matching is not None:
                    matches.append((matching.group(0), identifier))
            if matches:
                # get the longest match
                lexeme, identifier = max(matches, key=lambda t: len(t[0]))
                self._skip_n_chars(len(lexeme) - 1)
                if lexeme in self.reserved:
                    token = Terminal.from_token_type(lexeme, token_location)
                else:
                    token = Terminal(identifier, lexeme, token_location)
            else:
                # we try to match whitespace while avoiding NEWLINES because we
                # are using NEWLINES to split lines in our program
                if self._current_char() != "\n" and self._current_char().isspace():
                    long_whitespace = re.match(
                        r"[ \r\t]+", self._remaining_code()
                    ).group(0)
                    token = Terminal("whitespace", long_whitespace, token_location)
                    self._skip_n_chars(len(long_whitespace) - 1)
                elif self._current_char() == "#":
                    token = self.handle_comment()
                elif self._current_char() == "\n":
                    token = Terminal.from_token_type("newline", token_location)
                    self._linenum += 1
                    # we set column to -1 because it will be incremented to 0 after the token has been yielded
                    self._column = -1
                else:
                    raise ValueError(
                        'unrecognized token: "' + self._current_char() + '"'
                    )

            yield token
            self._to_next_char()

        # we must always end our stream of tokens with an EOF token
        yield EOF

    def handle_comment(self):
        end_comment_pos = self._remaining_code().index("\n")
        if end_comment_pos == -1:
            raise ValueError()
        comment = self._remaining_code()[:end_comment_pos]
        token = Terminal(
            "comment",
            comment,
            Loc(
                self._filename,
                self._linenum,
                self._column,
                self._code_offset,
            ),
        )
        self._skip_n_chars(len(comment))
        self._linenum += 1
        # we set column to -1 because it will be incremented to 0 after the token has been yielded
        self._column = -1
        return token

    def get_tokens(self, code: str) -> Iterator[Terminal]:
        """
        :return: an iterator over the tokens
        """
        self._reset(code)
        yield from self._tokenize()

    def get_tokens_no_whitespace(self, code: str):
        return [
            token
            for token in self.get_tokens(code)
            if not (
                token.name
                in (
                    "whitespace",
                    "newline",
                    "comment",
                )
            )
        ]


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
    (1, "("): 15,
    (1, "F"): 10,
    (1, "T"): 6,
    (1, "E"): 4,
    (1, "number"): 31,
    (2, "eof"): -1,
    (3, "+"): 19,
    (3, ")"): ("E0", 0),
    (3, "eof"): ("E0", 0),
    (3, "E0"): 8,
    (4, ")"): ("E", 2),
    (4, "eof"): ("E", 2),
    (5, "+"): ("T0", 0),
    (5, ")"): ("T0", 0),
    (5, "eof"): ("T0", 0),
    (5, "T0"): 12,
    (5, "*"): 25,
    (6, "+"): ("T", 2),
    (6, ")"): ("T", 2),
    (6, "eof"): ("T", 2),
    (7, "("): 15,
    (7, "F"): 10,
    (7, "T"): 6,
    (7, "E"): 16,
    (7, "number"): 31,
    (8, ")"): 33,
    (9, "("): 15,
    (9, "F"): 10,
    (9, "T"): 20,
    (9, "number"): 31,
    (10, "+"): 19,
    (10, ")"): ("E0", 0),
    (10, "eof"): ("E0", 0),
    (10, "E0"): 22,
    (11, ")"): ("E0", 3),
    (11, "eof"): ("E0", 3),
    (12, "("): 15,
    (12, "F"): 26,
    (12, "number"): 31,
    (13, "+"): ("T0", 0),
    (13, ")"): ("T0", 0),
    (13, "eof"): ("T0", 0),
    (13, "T0"): 28,
    (13, "*"): 25,
    (14, "+"): ("T0", 3),
    (14, ")"): ("T0", 3),
    (14, "eof"): ("T0", 3),
    (15, "+"): ("F", 1),
    (15, ")"): ("F", 1),
    (15, "eof"): ("F", 1),
    (15, "*"): ("F", 1),
    (16, "+"): ("F", 3),
    (16, ")"): ("F", 3),
    (16, "eof"): ("F", 3),
    (16, "*"): ("F", 3),
}  # type: ignore

states: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # type: ignore

expected_tokens: dict[int, list[str]] = {
    1: ["(", "number"],
    2: ["eof"],
    3: ["+", ")", "eof"],
    4: [")", "eof"],
    5: ["+", ")", "eof", "*"],
    6: ["+", ")", "eof"],
    7: ["(", "number"],
    8: [")"],
    9: ["(", "number"],
    10: ["+", ")", "eof"],
    11: [")", "eof"],
    12: ["(", "number"],
    13: ["+", ")", "eof", "*"],
    14: ["+", ")", "eof"],
    15: ["+", ")", "eof", "*"],
    16: ["+", ")", "eof", "*"],
}  # type: ignore


def parse(input_str: str) -> ParseTree:
    tokens = Tokenizer().get_tokens_no_whitespace(input_str)
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
