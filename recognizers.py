from abc import ABC, abstractmethod
from collections import deque
from typing import cast

from cfg import CFG
from core import EMPTY, EOF, NonTerminal, Rule, Terminal
from earley import gen_early_sets
from lr import LR0ParsingTable, Shift, Reduce, Goto, Accept
from tokenizer import Token

MAX_ITERATIONS = 1000_000


class RecognizerError(Exception):
    ...


class Recognizer(ABC):
    def __init__(self, grammar: CFG):
        self.grammar = grammar

    @abstractmethod
    def recognizes(self, tokens: list[Token]) -> bool:
        ...


class BFSTopDownLeftmostRecognizer(Recognizer):
    def recognizes(self, tokens: list[Token]) -> bool:
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
    def recognizes(self, tokens: list[Token]) -> bool:
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
    def recognizes(self, tokens: list[Token]) -> bool:
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


class EarleyRecognizer(Recognizer):
    def recognizes(self, tokens: list[Token]) -> bool:
        earley_sets = gen_early_sets(self.grammar, tokens)
        # are complete (the fat dot is at the end),
        # have started at the beginning (state set 0),
        # have the same name that has been chosen at the beginning ("Sum").

        return any(
            item.dot == len(item.rule)
            and item.explicit_index == 0
            and item.name == self.grammar.start_symbol
            for item in earley_sets[-1]
        )


class LR0Recognizer(Recognizer):
    def recognizes(self, tokens: list[Token]) -> bool:
        lr0_parsing_table = LR0ParsingTable(self.grammar)
        stack, token_index = [lr0_parsing_table.states[0]], 0
        while stack:
            state = stack[-1]
            token = tokens[token_index]
            match lr0_parsing_table.get((state, token.id)):
                # Advance input one token; push state n on stack.
                case Shift(state):
                    stack.append(state)
                    token_index += 1
                case Reduce(lhs, rule):
                    n_symbols = len(rule)
                    stack = stack[:-n_symbols]
                    match lr0_parsing_table[(stack[-1], lhs.id)]:
                        case Goto(state):
                            stack.append(state)
                        case _:
                            raise SyntaxError(f"Unexpected {token.id} at {token.loc}")
                case Accept():
                    return True
                case _:
                    raise SyntaxError(f"Unexpected {token.id} at {token.loc}")
        raise SyntaxError(
            f"Syntax error at {tokens[token_index] if token_index < len(tokens) else EOF}"
        )


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr
    from parse_grammar import parse_grammar
    from tokenizer import Tokenizer

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
    #     "+": "+",
    #     ";": ";",
    #     "(": "(",
    #     ")": ")",
    # }
    #
    # g = """
    #     <S>
    #     <S> -> <E>
    #     <E> -> <T>; | <T> + <E>
    #     <T> -> (<E>) | integer
    # """
    table = {
        "+": "+",
        "-": "-",
        "*": "*",
        "a": "a",
    }

    g = """
        <S>
        <S> -> <A>
        <A> -> <A> + <A> | <A> − <A> | a
    """
    tks = Tokenizer("1 + (2 + 3;);", table).get_tokens_no_whitespace()

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))
    p = LR0ParsingTable(cfg)
    print_rich(pretty_repr(p.states))
    print_rich(pretty_repr(p))

    # p.draw_with_graphviz()
    print_rich(pretty_repr(LR0Recognizer(cfg).recognizes(tks)))
