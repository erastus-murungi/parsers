import pytest

from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR_LR0


@pytest.mark.parametrize("grammar", [GRAMMAR_LR0])
def test_lr1_recognize(grammar):
    g, table = GRAMMAR_LR0
    cfg = Grammar.from_str(g, table)
    assert recognize(cfg, "1;", table, recognizer="lr0")
    assert recognize(cfg, "1 + 1;", table, recognizer="lr0")
    assert recognize(cfg, "(1 + 1;);", table, recognizer="lr0")
