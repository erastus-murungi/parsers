import pytest

import earley
from grammar import Grammar
from recognizers import recognize
from utils.grammars import (
    GRAMMAR1,
    GRAMMAR_AMBIGUOUS_PLUS_MINUS,
    GRAMMAR_DECAF,
    GRAMMAR_REGEX,
    RESERVED_DECAF,
)


@pytest.mark.parametrize("grammar", [GRAMMAR1])
def test_earley_recognizes(grammar):
    cfg = Grammar.from_str(grammar)
    assert recognize(cfg, "1 + (2 + 3)", recognizer="earley")


@pytest.mark.parametrize("grammar", [GRAMMAR1])
def test_earley_does_not_recognize(grammar):
    cfg = Grammar.from_str(grammar)
    with pytest.raises(earley.earley.EarleyError):
        recognize(cfg, "1 + (2 + 3", recognizer="earley")


def test_ambiguous_1():
    cfg = Grammar.from_str(GRAMMAR_AMBIGUOUS_PLUS_MINUS)
    assert recognize(cfg, "a + a", recognizer="earley")
    assert recognize(cfg, "a - a", recognizer="earley")
    assert recognize(cfg, "a + a - a", recognizer="earley")


def test_earley_on_regex():
    cfg = Grammar.from_str(GRAMMAR_REGEX)
    assert recognize(cfg, "a", recognizer="earley")
    assert recognize(cfg, "a*", recognizer="earley")
    assert recognize(cfg, "a|b*", recognizer="earley")
    assert recognize(cfg, "a|b+", recognizer="earley")
    assert recognize(cfg, "^a+(?:ab)[a-z]\\w+c{1,3}$", recognizer="earley")
    assert recognize(cfg, "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p", recognizer="earley")
    assert recognize(cfg, "^$", recognizer="earley")


def test_earley_on_decaf():
    cfg = Grammar.from_str(GRAMMAR_DECAF, RESERVED_DECAF)
    prog = """
        import java;
        
        void main (int a, int b, float c) {
            int a, b, i;
            a = 10;
            b = 20;
            
            if (a > b) {
                a = b;
            } else {
                b = a;
            }
            for (int i = 0; i < 10; i++) {
                a = a + 1;
            }
            return a + b;
        }
    """
    assert recognize(cfg, prog, recognizer="earley")
