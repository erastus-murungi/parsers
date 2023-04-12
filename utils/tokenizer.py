import itertools
import re
from dataclasses import dataclass
from typing import Iterator, NamedTuple, Optional


def group(*choices):
    return "(" + "|".join(choices) + ")"


def reg_any(*choices):
    return group(*choices) + "*"


def maybe(*choices):
    return group(*choices) + "?"


# Regular expressions used to parse numbers
Hexnumber = r"0[xX](?:_?[0-9a-fA-F])+"
Binnumber = r"0[bB](?:_?[01])+"
Octnumber = r"0[oO](?:_?[0-7])+"
Decnumber = r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)"
Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
Pointfloat = group(
    r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
) + maybe(Exponent)
Expfloat = r"[0-9](?:_?[0-9])*" + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group(r"[0-9](?:_?[0-9])*[jJ]", Floatnumber + r"[jJ]")


@dataclass
class Token:
    """
    A token has three components:
    1) Its type
    2) A lexeme -- the substring of the source code it represents
    3) The location in code of the lexeme
    """

    token_type: str
    lexeme: str
    loc: "Tokenizer.Loc"

    @staticmethod
    def from_token_type(token_type: str, loc: "Tokenizer.Loc"):
        """A convenient constructor to avoid the frequent pattern:
        Token(TokenType.X, TokenType.X.value, loc)"""
        return Token(token_type, token_type, loc)

    @property
    def id(self) -> str:
        return self.token_type


class Tokenizer:
    class Loc(NamedTuple):
        filename: str
        line: int
        col: int
        offset: int

        def __str__(self):
            return f"<{self.filename}:{self.line}:{self.col}>"

    def __init__(
        self, code: str, _named_tokens: dict[str, str], filename: str = "(void)"
    ):
        self._filename = filename
        self._named_tokens = _named_tokens
        self._code = code + "\n"
        self._linenum = 0
        self._column = 0
        self._code_offset = 0
        self._token_iterable = self._tokenize()

    def _to_next_char(self):
        self._code_offset += 1
        self._column += 1

    def _skip_n_chars(self, n):
        self._code_offset += n
        self._column += n

    def _match_number(self, number_regex: str) -> Optional[tuple[str, str]]:
        match = re.match("^" + number_regex, self._code[self._code_offset :])
        token_type = None
        lexeme = ""
        if match is not None:
            lexeme, token_type = (
                match.group(0),
                "float"
                if number_regex == Floatnumber
                else "integer"
                if number_regex == Intnumber
                else None,
            )
        if token_type is None:
            return None
        else:
            return lexeme, token_type

    def _match_number_or_unknown(self, pos) -> Token:
        ret = (
            self._match_number(Imagnumber)
            or self._match_number(Floatnumber)
            or self._match_number(Intnumber)
        )
        if ret is None:
            # no token found
            return Token("char", self._current_char(), pos)

        lexeme, ret_type = ret
        self._skip_n_chars(len(lexeme) - 1)
        return Token(ret_type, lexeme, pos)

    def _current_char(self):
        return self._code[self._code_offset]

    def _remaining_code(self):
        return self._code[self._code_offset :]

    def _tokenize(self) -> Iterator[Token]:
        while self._code_offset < len(self._code):
            token_location = self.Loc(
                self._filename, self._linenum, self._column, self._code_offset
            )
            # greedy attempt
            for matching, identifier in sorted(
                self._named_tokens.items(), key=lambda item: len(item[0])
            ):
                if self._remaining_code().startswith(matching):
                    # this is a keyword
                    self._skip_n_chars(len(matching) - 1)
                    token = Token(identifier, matching, token_location)
                    break
            else:
                # we try to match whitespace while avoiding NEWLINES because we
                # are using NEWLINES to split lines in our program
                if self._current_char() != "\n" and self._current_char().isspace():
                    token = Token("whitespace", self._current_char(), token_location)
                elif self._current_char() == "#":
                    token = self.handle_comment()
                elif self._current_char() == "\n":
                    token = Token.from_token_type("newline", token_location)
                    self._linenum += 1
                    # we set column to -1 because it will be incremented to 0 after the token has been yielded
                    self._column = -1
                else:
                    token = self._match_number_or_unknown(token_location)

            yield token
            self._to_next_char()

        # we must always end our stream of tokens with an EOF token
        yield Token(
            "eof",
            "$",
            self.Loc(self._filename, self._linenum, self._column, self._code_offset),
        )

    def handle_comment(self):
        end_comment_pos = self._remaining_code().index("\n")
        if end_comment_pos == -1:
            raise ValueError()
        comment = self._remaining_code()[:end_comment_pos]
        token = Token(
            "comment",
            comment,
            self.Loc(
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

    def get_tokens(self) -> Iterator[Token]:
        """
        :return: an iterator over the tokens
        """
        t1, t2 = itertools.tee(self._token_iterable)
        self._token_iterable = t1
        return t2

    def get_tokens_no_whitespace(self):
        return [
            token
            for token in self.get_tokens()
            if not (
                token.token_type
                in (
                    "whitespace",
                    "newline",
                    "comment",
                )
            )
        ]
