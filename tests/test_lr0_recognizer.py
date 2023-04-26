import pytest

from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR_LR0


@pytest.mark.parametrize("grammar", [GRAMMAR_LR0])
def test_lr1_recognize(grammar):
    cfg = Grammar.from_str(grammar)
    assert recognize(cfg, "1;", recognizer="lr0")
    assert recognize(cfg, "1 + 1;", recognizer="lr0")
    assert recognize(cfg, "(1 + 1;);", recognizer="lr0")
