from abc import ABC, abstractmethod
from collections import deque
from typing import Literal, cast

from earley import gen_earley_sets
from grammar import EMPTY, EOF, Expansion, Grammar, NonTerminal, Terminal
from ll import LL1ParsingTable
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
from tokenizer import Tokenizer

MAX_ITERATIONS = 1000_000


class RecognizerError(Exception):
    ...


class Recognizer(ABC):
    def __init__(self, grammar: Grammar, source: str, table):
        self.grammar: Grammar = grammar
        self.source: str = source
        self.tokens: list[Terminal] = Tokenizer(
            source, table
        ).get_tokens_no_whitespace()

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

            for index, symbol in rule.enumerate_variables():
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


class DFSTopDownLeftmostRecognizer(Recognizer):
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
            for index, symbol in rule.enumerate_variables():
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


class LL1Recognizer(Recognizer):
    def recognizes(self) -> bool:
        parsing_table = LL1ParsingTable(self.grammar)
        stack, token_index = [EOF, self.grammar.start], 0

        while stack:
            symbol = stack.pop()
            token = self.tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
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


class EarleyRecognizer(Recognizer):
    def recognizes(self) -> bool:
        gen_earley_sets(self.grammar, self.tokens, self.source)
        return True


class LR0Recognizer(Recognizer):
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


class SLRRecognizer(LR0Recognizer):
    def get_parsing_table(self):
        return SLRParsingTable(self.grammar)


class LR1Recognizer(LR0Recognizer):
    def get_parsing_table(self):
        return LR1ParsingTable(self.grammar)


class LALR1Recognizer(LR0Recognizer):
    def get_parsing_table(self):
        return LALR1ParsingTable(self.grammar)


class CYKRecognizer(Recognizer):
    pass


def recognize(
    grammar: Grammar,
    source: str,
    table: dict,
    *,
    recognizer: Literal["earley", "lalr1", "ll1", "slr", "lr1", "lr0", "dfs"],
) -> bool:
    match recognizer:
        case "earley":
            return EarleyRecognizer(grammar, source, table).recognizes()
        case "lalr1":
            return LALR1Recognizer(grammar, source, table).recognizes()
        case "ll1":
            return LL1Recognizer(grammar, source, table).recognizes()
        case "slr":
            return SLRRecognizer(grammar, source, table).recognizes()
        case "lr1":
            return LR1Recognizer(grammar, source, table).recognizes()
        case "lr0":
            return LR0Recognizer(grammar, source, table).recognizes()
        case "dfs":
            return DFSTopDownLeftmostRecognizer(grammar, source, table).recognizes()
        case _:
            raise ValueError(f"Unknown recognizer {recognizer}")


if __name__ == "__main__":
    pass

    # table = {
    #     "x": "x",
    #     "(": "(",
    #     ")": ")",
    #     ",": ",",
    # }
    # g = """
    #         <S'>
    #         <S'> -> <S>
    #         <S> -> (<L>)
    #         <L> -> <S>
    #         <S> -> x
    #         <L> -> <L>,<S>
    # """
    # table = {
    #     "a": "a",
    #     "b": "b",
    #     "c": "c",
    #     "d": "d",
    # }
    #
    # g = """
    # <S'>
    # <S'> -> <S>
    # <S> -> a <A> d | b <B> d | a <B> e | b <A> e
    # <A> -> c
    # <B> -> c
    # """
    #
    # g = """
    #     <S>
    #     <S> -> <E>
    #     <E> -> <T> | <T> + <E>
    #     <T> -> (<E>) | integer
    # """
    # table = {
    #     "+": "+",
    #     ";": ";",
    #     "(": "(",
    #     ")": ")",
    #     "=": "=",
    #     "*": "*",
    # }
    #
    # g = """
    #     <S>
    #     <S> -> <E>
    #     <E> -> <L> = <R> | <R>
    #     <L> -> char | *<R>
    #     <R> -> <L>
    # """
