import pytest

import earley
from recognizers import recognize
from utils.grammars import GRAMMAR1, GRAMMAR_AMBIGUOUS_PLUS_MINUS
from utils.parse_grammar import parse_grammar


@pytest.mark.parametrize("grammar", [GRAMMAR1])
def test_earley_recognizes(grammar):
    table, g = grammar
    cfg = parse_grammar(g, table)
    assert recognize(cfg, "1 + (2 + 3)", table, recognizer="earley")


@pytest.mark.parametrize("grammar", [GRAMMAR1])
def test_earley_does_not_recognize(grammar):
    table, g = grammar
    cfg = parse_grammar(g, table)
    with pytest.raises(earley.earley.EarleyError):
        recognize(cfg, "1 + (2 + 3", table, recognizer="earley")


def test_ambiguous_1():
    table, g = GRAMMAR_AMBIGUOUS_PLUS_MINUS
    cfg = parse_grammar(g, table)
    assert recognize(cfg, "a + a", table, recognizer="earley")
    assert recognize(cfg, "a - a", table, recognizer="earley")
    assert recognize(cfg, "a + a - a", table, recognizer="earley")
