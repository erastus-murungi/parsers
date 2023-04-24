import re
from collections import deque
from itertools import count
from typing import Iterator

from more_itertools import sliced

from grammar import EMPTY, Grammar, Loc, NonTerminal, Symbol, Terminal

temps_counter = count(0)
dummy_loc = Loc("", -1, -1, -1)


def iter_symbol_tokens(input_str: str) -> Iterator[str]:
    input_str = input_str.strip()
    while input_str:
        if m := re.match(r"^\|", input_str):
            yield m.group(0)
        elif m := re.match(r"<\w+>[?*+]?", input_str):  # NonTerminal
            yield m.group(0)
        elif m := re.match(r"((?<!')\(.*\)(?!')[?*+]?)", input_str):  # Grouped items
            yield m.group(0)
        elif m := re.match(r"'\w+?'[?*+]?", input_str):  # 'any word literal'
            yield m.group(0)
        elif m := re.match(r"\w+", input_str):  # keyword
            yield m.group(0)
        elif m := re.match(r"'.*?'[?*+]?", input_str):  # any literal
            yield m.group(0)
        else:
            raise ValueError(f"Invalid token: {input_str}")
        input_str = input_str[m.end() :].strip()


def parse_grammar(grammar_str: str, defined_tokens: dict[str, str]) -> Grammar:
    """Ad Hoc grammar parser"""
    grammar_builder = Grammar.Builder()
    for origin_str, definition_str in sliced(
        re.split(r"<(\w+)>\s*->", grammar_str.strip())[1:], n=2, strict=True
    ):
        origin = NonTerminal(origin_str)
        queue = deque([(origin, definition_str)])
        while queue:
            origin, expansion = queue.popleft()
            if isinstance(expansion, str):
                rule: list[Symbol] = []
                for lexeme in iter_symbol_tokens(expansion):
                    if lexeme == "|":
                        queue.append((origin, rule if rule else (EMPTY,)))
                        rule = []
                    elif (
                        lexeme.endswith("?")
                        | lexeme.endswith("*")
                        | lexeme.endswith("+")
                    ):
                        if lexeme.startswith("("):
                            R = NonTerminal(f"R_{next(temps_counter)}")
                            queue.append((R, lexeme[1:-2]))
                        elif lexeme.startswith("'"):
                            R = NonTerminal(f"R_{next(temps_counter)}")
                            queue.append((R, lexeme[:-1]))
                        else:
                            assert lexeme.startswith("<")
                            R = NonTerminal(lexeme[1:-2])

                        N = NonTerminal(f"N_{next(temps_counter)}")
                        if lexeme.endswith("?"):
                            # R? ⇒    N → ε
                            #         N → R
                            queue.append((N, (EMPTY,)))
                            queue.append((N, (R,)))
                        elif lexeme.endswith("*"):
                            # R* ⇒    N → ε
                            #         N → N R
                            queue.append((N, (EMPTY,)))
                            queue.append((N, (N, R)))
                        else:
                            # R+ ⇒    N → R
                            #         N → N R
                            assert lexeme.endswith("+")
                            queue.append((N, (R,)))
                            queue.append((N, (N, R)))
                        rule.append(N)

                    elif lexeme == "<>":
                        rule.append(EMPTY)
                    elif lexeme.startswith(r"\\"):
                        # this is a terminal
                        rule.append(Terminal(lexeme[1:], lexeme[1:], dummy_loc))
                    elif lexeme.startswith("<"):
                        # this is a non-terminal
                        rule.append(NonTerminal(lexeme[1:-1]))
                    elif lexeme in (
                        "integer",
                        "float",
                        "whitespace",
                        "newline",
                        "char",
                        "word",
                    ):
                        # keywords
                        rule.append(Terminal(lexeme, lexeme, dummy_loc))
                    elif lexeme in defined_tokens or re.match(r"'.*'", lexeme):
                        lexeme = lexeme[1:-1]
                        rule.append(
                            Terminal(
                                defined_tokens.get(lexeme, lexeme),
                                lexeme,
                                dummy_loc,
                            )
                        )
                    else:
                        raise ValueError(f"unknown symbol {lexeme}\n{expansion}")
                grammar_builder.add_expansion(origin, rule)
            else:
                grammar_builder.add_expansion(origin, expansion)

    return grammar_builder.build()
