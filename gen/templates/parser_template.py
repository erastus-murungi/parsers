"""
    ID: %id%
"""

import re
from typing import Iterator

from grammar import Loc
from more_itertools import one

from grammar import EOF, Terminal
from parsers.parser import ParseTree


patterns: dict[str, re.Pattern] = "%patterns%"


class Tokenizer:
    def __init__(self, filename: str = "%filename%"):
        self._filename = filename
        self._code = ""
        self._linenum = 0
        self._column = 0
        self._code_offset = 0
        self._named_tokens = patterns

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

    def _match_unknown(self, pos) -> Terminal:
        word_match = re.match(r"\b\w+\b", self._remaining_code())
        if word_match is not None and len(word_match.group(0)) > 1:
            word = word_match.group(0).strip()
            lexeme, ret_type = word, "word"
        else:
            lexeme, ret_type = self._current_char(), "char"
        self._skip_n_chars(len(lexeme) - 1)
        return Terminal(ret_type, lexeme, pos)

    def _current_char(self):
        return self._code[self._code_offset]

    def _remaining_code(self):
        return self._code[self._code_offset :]

    def tokenize(self) -> Iterator[Terminal]:
        while self._code_offset < len(self._code):
            token_location = Loc(
                self._filename, self._linenum, self._column, self._code_offset
            )
            matches: list[tuple[str, str]] = []
            for identifier, pattern in self._named_tokens.items():
                matching = pattern.match(self._remaining_code())
                if matching is not None:
                    matches.append((matching.group(0), identifier))
            if matches:
                lexeme, identifier = max(matches, key=lambda t: len(t[0]))
                self._skip_n_chars(len(lexeme) - 1)
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
                    token = self._match_unknown(token_location)

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
        yield from self.tokenize()

    def get_tokens_no_whitespace(self, code: str):
        return [
            token
            for token in self.get_tokens(code)
            if not (
                token.token_type
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


parsing_table: dict[tuple[int, str], Action] = "%parsing_table%"  # type: ignore

states: list[int] = "%states%"  # type: ignore

expected_tokens: dict[int, list[str]] = "%expected_tokens%"  # type: ignore


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
