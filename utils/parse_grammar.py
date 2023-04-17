import re
from typing import Final, Iterator

from grammar import EMPTY, Grammar, NonTerminal, Symbol, Terminal

NON_TERMINAL_REGEX: Final[str] = r"<([\w\']+)>"
SEPARATOR = r"->"


def iter_symbol_tokens(input_str: str) -> Iterator[str]:
    input_str = input_str.strip()
    start_index = 0
    for non_terminal_match in re.finditer(NON_TERMINAL_REGEX, input_str):
        yield from input_str[start_index : non_terminal_match.start()].split()
        yield non_terminal_match.group(0)
        start_index = non_terminal_match.end()
    yield from input_str[start_index:].split()


def bind_lexeme(lexeme: str):
    return lambda token: token.lexeme == lexeme


def bind_token_type(token_type: str):
    return lambda token: token.token_type == token_type


def parse_grammar(grammar_str: str, defined_tokens: dict[str, str]) -> Grammar:
    """Ad Hoc grammar parser"""

    definitions = grammar_str.strip().split("\n")

    cfg = Grammar.Builder()

    for definition in definitions:
        if not definition.strip():
            continue

        lhs_str, rhs_str = re.split(SEPARATOR, definition)

        if (lhs_match := re.match(NON_TERMINAL_REGEX, lhs_str.strip())) is None:
            raise ValueError(
                f"no non-terminal on rhs of {definition}, check that syntax is correct"
            )
        lhs = NonTerminal(lhs_match.group(1))

        for rule_str in rhs_str.split("|"):

            rule: list[Symbol] = []

            for lexeme in iter_symbol_tokens(rule_str):
                if lexeme == "<>":
                    rule.append(EMPTY)
                elif lexeme.startswith(r"\\"):
                    # this is a terminal
                    rule.append(Terminal(lexeme[1:], bind_lexeme(lexeme[1:])))
                elif lexeme.startswith("<"):
                    # this is a non-terminal
                    rule.append(NonTerminal(lexeme[1:-1]))
                elif lexeme in ("integer", "float", "whitespace", "newline", "char"):
                    # keywords
                    rule.append(Terminal(lexeme, bind_token_type(lexeme)))
                else:
                    rule.append(
                        Terminal(
                            defined_tokens.get(lexeme, lexeme),
                            bind_lexeme(lexeme),
                        )
                    )
            cfg.add_expansion(lhs, rule)

    return cfg.build()
