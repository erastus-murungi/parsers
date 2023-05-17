from abc import ABC, abstractmethod
from collections import deque
from typing import Literal, cast

from cyk.cyk import cyk_parse
from earley import gen_earley_sets
from grammar import EMPTY, EOF, Expansion, Grammar, NonTerminal, Terminal
from ll import LL1ParsingTable
from ll.core import TerminalSequence
from ll.llk import LLKParsingTable
from lr import (
    Accept,
    Goto,
    LALR1ParsingTable,
    LR0ParsingTable,
    LR1ParsingTable,
    Reduce,
    Shift,
    SLRParsingTable,
)

MAX_ITERATIONS = 1000_000


class RecognizerError(Exception):
    ...


class Recognizer(ABC):
    def __init__(self, grammar: Grammar, source: str):
        self.grammar: Grammar = grammar
        self.source: str = source
        self.tokens: list[Terminal] = grammar.tokenizer.get_tokens_no_whitespace(source)

    @abstractmethod
    def recognizes(self) -> bool:
        ...


class BFSTopDownLeftmostRecognizer(Recognizer):
    def recognizes(self) -> bool:
        rules: deque[Expansion] = deque([Expansion([self.grammar.start, EOF])])
        seen: set[Expansion] = set()
        nullable_set = self.grammar.gen_nullable()

        n_iters = 0
        while rules and n_iters < MAX_ITERATIONS:
            if (rule := rules.popleft()).matches(self.tokens):
                return True

            seen.add(rule)

            for index, symbol in rule.enumerate_non_terminals():
                for replacement in self.grammar[symbol]:
                    if (
                        next_form := rule.perform_derivation(index, replacement)
                    ).should_prune(self.tokens, seen, nullable_set):
                        continue
                    rules.append(next_form)

            n_iters += 1

        if n_iters >= MAX_ITERATIONS:
            raise RecognizerError("Too many iterations")
        else:
            raise RecognizerError("No rules left to explore")


class DfsRecognizer(Recognizer):
    def recognizes(self) -> bool:
        rules: list[Expansion] = [Expansion([self.grammar.start, EOF])]
        seen: set[Expansion] = set()
        nullable_set = self.grammar.gen_nullable()

        n_iters = 0
        while rules and n_iters < MAX_ITERATIONS:
            if (rule := rules.pop()).matches(self.tokens):
                return True

            seen.add(rule)

            next_in_stack = []
            for index, symbol in rule.enumerate_non_terminals():
                for replacement in self.grammar[symbol]:
                    if (
                        next_form := rule.perform_derivation(index, replacement)
                    ).should_prune(self.tokens, seen, nullable_set):
                        continue

                    next_in_stack.append(next_form)
            rules.extend(reversed(next_in_stack))

            n_iters += 1

        if n_iters >= MAX_ITERATIONS:
            raise RecognizerError("Too many iterations")
        else:
            raise RecognizerError("No rules left to explore")


class Ll1Recognizer(Recognizer):
    def recognizes(self) -> bool:
        parsing_table = LL1ParsingTable(self.grammar)
        stack, token_index = [EOF, self.grammar.orig_start], 0

        while stack:
            symbol = stack.pop()
            token = self.tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol == token:
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.name} but got {token}")
            else:
                non_terminal = cast(NonTerminal, symbol)
                if (rule := parsing_table.get((non_terminal, token.name))) is not None:
                    stack.extend(reversed(rule))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.name for terminal in self.grammar.gen_first()[symbol])}), '
                        f"but found {token.name!s}"
                    )
        assert token_index >= len(self.tokens)
        return True


class LlkRecognizer(Recognizer):
    def recognizes(self) -> bool:
        parsing_table = LLKParsingTable(self.grammar)
        stack, token_index = [EOF, self.grammar.orig_start], 0

        while stack:
            symbol = stack.pop()
            token = self.tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol == token:
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.name} but got {token}")
            else:
                # choose a rule depending on the next k tokens
                non_terminal = cast(NonTerminal, symbol)
                if (
                    rule := parsing_table.choose_rule(
                        non_terminal, token_index, self.tokens
                    )
                ) is not None:
                    stack.extend(reversed(rule))
                else:
                    found = [
                        str(
                            TerminalSequence(
                                self.tokens[token_index : token_index + i], i
                            )
                        )
                        for i in range(1, parsing_table.k + 1)
                    ]
                    raise SyntaxError(
                        f"At position {token.loc}\n "
                        f">  {self.source[token.loc.offset: token.loc.offset + 40]}\n"
                        f"tokens: {[tk.lexeme for tk in self.tokens[token_index: token_index + parsing_table.k]]}\n"
                        f"was parsing {symbol!s} "
                        f"expecting one of :\n"
                        f"\t\t{parsing_table.get_expected(non_terminal)}\n"
                        f"but found {found!s}"
                    )
        assert token_index >= len(self.tokens)
        return True


class EarleyRecognizer(Recognizer):
    def recognizes(self) -> bool:
        gen_earley_sets(self.grammar, self.tokens, self.source)
        return True


class Lr0Recognizer(Recognizer):
    def get_parsing_table(self):
        return LR0ParsingTable(self.grammar)

    def recognizes(self) -> bool:
        parsing_table = self.get_parsing_table()
        stack, token_index = [parsing_table.states[0]], 0
        while stack:
            current_state = stack[-1]
            current_token = self.tokens[token_index]
            match parsing_table.get((current_state, current_token.name)):
                # Advance input one token; push state n on stack.
                # TODO: assert that current_state corresponds to the current_token
                case Shift(current_state):
                    stack.append(current_state)
                    token_index += current_token.name != EOF.name
                case Reduce(lhs, len_rhs):
                    stack = stack[: -len_rhs or None]
                    match parsing_table[(stack[-1], lhs.name)]:
                        case Goto(current_state):
                            stack.append(current_state)
                        case _:
                            raise SyntaxError(
                                f"Unexpected {current_token.name} at {current_token.loc}"
                            )
                case Accept():
                    return True
                case _:
                    raise SyntaxError(
                        f"Unexpected {current_token.name} at {current_token.loc}"
                    )
        raise SyntaxError(
            f"Syntax error at {self.tokens[token_index] if token_index < len(self.tokens) else EOF}"
        )


class SlrRecognizer(Lr0Recognizer):
    def get_parsing_table(self):
        return SLRParsingTable(self.grammar)


class Lr1Recognizer(Lr0Recognizer):
    def get_parsing_table(self):
        return LR1ParsingTable(self.grammar)


class Lalr1Recognizer(Lr0Recognizer):
    def get_parsing_table(self):
        return LALR1ParsingTable(self.grammar)


class CykRecognizer(Recognizer):
    def recognizes(self) -> bool:
        _, table, _ = cyk_parse(self.grammar, self.source)
        return self.grammar.orig_start in table[(0, len(self.tokens) - 2)]


def recognize(
    grammar: Grammar,
    source: str,
    *,
    recognizer: Literal[
        "earley", "lalr1", "ll1", "slr", "lr1", "lr0", "dfs", "llk", "cyk"
    ],
) -> bool:
    cls = globals().get(f"{recognizer.capitalize()}Recognizer")
    return cls(grammar, source).recognizes()


if __name__ == "__main__":
    from rich import print as rich_print
    from rich.pretty import pretty_repr

    from utils.grammars import GRAMMAR_JSON

    g = Grammar.from_str(GRAMMAR_JSON, transform_regex_to_right=False)
    rich_print(pretty_repr(g))
    rich_print(pretty_repr(recognize(g, "[1, 2, 4, 5]", recognizer="cyk")))
