from abc import ABC, abstractmethod
from collections import deque
from typing import cast

from general import gen_early_sets
from grammar import CFG
from grammar.core import EMPTY, EOF, NonTerminal, Rule, Terminal
from lalr.lalr1 import LALR1ParsingTable
from ll.ll1 import LL1ParsingTable
from lr.core import Accept, Goto, Reduce, Shift
from lr.lr1 import LR1ParsingTable
from lr.slr import LR0ParsingTable, SLRParsingTable
from utils.tokenizer import Token

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
        parsing_table = LL1ParsingTable(self.grammar)
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
    def get_parsing_table(self):
        return LR0ParsingTable(self.grammar)

    def recognizes(self, tokens: list[Token]) -> bool:
        parsing_table = self.get_parsing_table()
        stack, token_index = [parsing_table.states[0]], 0
        while stack:
            current_state = stack[-1]
            current_token = tokens[token_index]
            match parsing_table.get((current_state, current_token.id)):
                # Advance input one token; push state n on stack.
                # TODO: assert that current_state corresponds to the current_token
                case Shift(current_state):
                    stack.append(current_state)
                    token_index += current_token.id != EOF.id
                case Reduce(lhs, rule):
                    stack = stack[: -len(rule)]
                    match parsing_table[(stack[-1], lhs.id)]:
                        case Goto(current_state):
                            stack.append(current_state)
                        case _:
                            raise SyntaxError(
                                f"Unexpected {current_token.id} at {current_token.loc}"
                            )
                case Accept():
                    return True
                case _:
                    raise SyntaxError(
                        f"Unexpected {current_token.id} at {current_token.loc}"
                    )
        raise SyntaxError(
            f"Syntax error at {tokens[token_index] if token_index < len(tokens) else EOF}"
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


if __name__ == "__main__":
    from rich import print as print_rich
    from rich.pretty import pretty_repr

    from utils.parse_grammar import parse_grammar

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
    #     <E> -> <T>; | <T> + <E>
    #     <T> -> (<E>) | integer
    # """
    # g = """
    #     <S>
    #     <S> -> <E>
    #     <E> -> <T> | <T> + <E>
    #     <T> -> (<E>) | integer
    # """

    table = {
        "+": "+",
        ";": ";",
        "(": "(",
        ")": ")",
        "=": "=",
        "*": "*",
    }

    g = """
        <S>
        <S> -> <E>
        <E> -> <L> = <R> | <R>
        <L> -> char | *<R>
        <R> -> <L>
    """

    # table = {
    #     "+": "+",
    #     "-": "-",
    #     "*": "*",
    #     "a": "a",
    # }
    #
    # g = """
    #     <S>
    #     <S> -> <A>
    #     <A> -> <A> + <A> | <A> − <A> | a
    # """

    cfg = parse_grammar(g, table)
    print_rich(pretty_repr(cfg))
    p = LR1ParsingTable(cfg)
    print_rich(p.to_pretty_table())
    p1 = LALR1ParsingTable(cfg)
    print_rich(p1.to_pretty_table())

    # p.draw_with_graphviz()
    # tks = Tokenizer("1 + (2 + 3)", table).get_tokens_no_whitespace()
    # print_rich(pretty_repr(LR1Recognizer(cfg).recognizes(tks)))
