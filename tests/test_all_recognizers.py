import pytest

from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR_DYCK


@pytest.mark.parametrize("grammar", [GRAMMAR_DYCK])
def test_can_parse_dyck(grammar):
    g, table = GRAMMAR_DYCK
    cfg = Grammar.from_str(g, table)
    assert recognize(cfg, "()(())", table, recognizer="lr1")
    assert recognize(cfg, "((()))", table, recognizer="slr")
    assert recognize(cfg, "((()))", table, recognizer="earley")
    assert recognize(cfg, "((()))", table, recognizer="ll1")
    assert recognize(cfg, "((()))", table, recognizer="dfs")
