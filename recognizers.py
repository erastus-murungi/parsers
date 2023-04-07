from abc import ABC, abstractmethod
from collections import deque

from cfg import CFG
from core import EOF, Rule
from tokenizer import Tokenizer

MAX_ITERATIONS = 1000_000


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

        return False


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

        return False
