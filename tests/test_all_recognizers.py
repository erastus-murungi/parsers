import pytest

from recognizers import recognize
from utils.grammars import GRAMMAR_DYCK
from utils.parse_grammar import parse_grammar


@pytest.mark.parametrize("grammar", [GRAMMAR_DYCK])
def test_can_parse_dyck(grammar):
    table, g = GRAMMAR_DYCK
    cfg = parse_grammar(g, table)
    assert recognize(cfg, "()(())", table, recognizer="lr1")
    assert recognize(cfg, "((()))", table, recognizer="slr")
    assert recognize(cfg, "((()))", table, recognizer="earley")
    assert recognize(cfg, "((()))", table, recognizer="ll1")
    assert recognize(cfg, "((()))", table, recognizer="dfs")
