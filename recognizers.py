from abc import ABC, abstractmethod
from collections import deque
from typing import cast

from cfg import CFG
from core import EMPTY, EOF, NonTerminal, Rule, Terminal
from tokenizer import Tokenizer

MAX_ITERATIONS = 1000_000


class RecognizerError(Exception):
    ...


class Recognizer(ABC):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    @abstractmethod
    def recognizes(self, tokens: list[Tokenizer.Token]) -> bool:
        ...


class BFSTopDownLeftmostRecognizer(Recognizer):
    def recognizes(self, tokens: list[Tokenizer.Token]) -> bool:
        rules: deque[Rule] = deque([Rule([self.grammar.start_symbol, EOF])])
        seen: set[Rule] = set()
        nullable_set = self.grammar.nullable()

        n_iters = 0
        while rules and n_iters < MAX_ITERATIONS:
            if (rule := rules.popleft()).matches(tokens):
                return True

            seen.add(rule)

            for index, symbol in rule.enumerate_variables():
                for replacement in self.grammar[symbol]:
                    if (
                        next_form := rule.perform_derivation(index, replacement)
                    ).should_prune(tokens, seen, nullable_set):
                        continue
                    rules.append(next_form)

            n_iters += 1

        if n_iters >= MAX_ITERATIONS:
            raise RecognizerError("Too many iterations")
        else:
            raise RecognizerError("No rules left to explore")


class DFSTopDownLeftmostRecognizer(Recognizer):
    def recognizes(self, tokens: list[Tokenizer.Token]) -> bool:
        rules: list[Rule] = [Rule([self.grammar.start_symbol, EOF])]
        seen: set[Rule] = set()
        nullable_set = self.grammar.nullable()

        n_iters = 0
        while rules and n_iters < MAX_ITERATIONS:
            if (rule := rules.pop()).matches(tokens):
                return True

            seen.add(rule)

            next_in_stack = []
            for index, symbol in rule.enumerate_variables():
                for replacement in self.grammar[symbol]:
                    if (
                        next_form := rule.perform_derivation(index, replacement)
                    ).should_prune(tokens, seen, nullable_set):
                        continue

                    next_in_stack.append(next_form)
            rules.extend(reversed(next_in_stack))

            n_iters += 1

        if n_iters >= MAX_ITERATIONS:
            raise RecognizerError("Too many iterations")
        else:
            raise RecognizerError("No rules left to explore")


class LL1Recognizer(Recognizer):
    def recognizes(self, tokens: list[Tokenizer.Token]) -> bool:
        parsing_table = self.grammar.build_ll1_parsing_table()
        stack, token_index = [EOF, self.grammar.start_symbol], 0

        while stack:
            symbol = stack.pop()
            token = tokens[token_index]
            if isinstance(symbol, Terminal):
                if symbol.matches(token):
                    token_index += symbol is not EMPTY
                else:
                    raise SyntaxError(f"Expected {symbol.id} but got {token}")
            else:
                non_terminal = cast(NonTerminal, symbol)
                if (rule := parsing_table.get((non_terminal, token.id))) is not None:
                    stack.extend(reversed(rule))
                else:
                    raise SyntaxError(
                        f"At position {token.loc}, "
                        f"was parsing {symbol!s} "
                        f'expecting one of ({", ".join(terminal.id for terminal in self.grammar.first()[symbol])}), '
                        f"but found {token.id!s}"
                    )
        assert token_index >= len(tokens)
        return True
