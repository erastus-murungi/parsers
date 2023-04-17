import pytest

from recognizers import recognize
from utils.grammars import GRAMMAR_LR0
from utils.parse_grammar import parse_grammar


@pytest.mark.parametrize("grammar", [GRAMMAR_LR0])
def test_lr1_recognize(grammar):
    table, g = GRAMMAR_LR0
    cfg = parse_grammar(g, table)
    assert recognize(cfg, "1;", table, recognizer="lr0")
    assert recognize(cfg, "1 + 1;", table, recognizer="lr0")
    assert recognize(cfg, "(1 + 1;);", table, recognizer="lr0")
