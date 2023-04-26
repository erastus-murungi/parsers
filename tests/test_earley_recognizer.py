import pytest

import earley
from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR1, GRAMMAR_AMBIGUOUS_PLUS_MINUS


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
