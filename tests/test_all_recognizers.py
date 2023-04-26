import pytest

from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR_DYCK


@pytest.mark.parametrize("grammar", [GRAMMAR_DYCK])
def test_can_parse_dyck(grammar):
    cfg = Grammar.from_str(GRAMMAR_DYCK)
    assert recognize(cfg, "()(())", recognizer="lr1")
    assert recognize(cfg, "((()))", recognizer="slr")
    assert recognize(cfg, "((()))", recognizer="earley")
    assert recognize(cfg, "((()))", recognizer="ll1")
    assert recognize(cfg, "((()))", recognizer="dfs")
    assert recognize(cfg, "((()))", recognizer="llk")
