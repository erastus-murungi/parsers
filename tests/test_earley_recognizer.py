import pytest

import earley
from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR1, GRAMMAR_AMBIGUOUS_PLUS_MINUS


@pytest.mark.parametrize("grammar", [GRAMMAR1])
def test_earley_recognizes(grammar):
    g, table = grammar
    cfg = Grammar.from_str(g, table)
    assert recognize(cfg, "1 + (2 + 3)", table, recognizer="earley")


@pytest.mark.parametrize("grammar", [GRAMMAR1])
def test_earley_does_not_recognize(grammar):
    g, table = grammar
    cfg = Grammar.from_str(g, table)
    with pytest.raises(earley.earley.EarleyError):
        recognize(cfg, "1 + (2 + 3", table, recognizer="earley")


def test_ambiguous_1():
    g, table = GRAMMAR_AMBIGUOUS_PLUS_MINUS
    cfg = Grammar.from_str(g, table)
    assert recognize(cfg, "a + a", table, recognizer="earley")
    assert recognize(cfg, "a - a", table, recognizer="earley")
    assert recognize(cfg, "a + a - a", table, recognizer="earley")
