import pytest

from recognizers import recognize
from utils.grammars import GRAMMAR1, GRAMMAR2
from utils.parse_grammar import parse_grammar


@pytest.mark.parametrize("grammar", [GRAMMAR1, GRAMMAR2])
def test_ll1_recognize(grammar):
    table, g = GRAMMAR1
    cfg = parse_grammar(g, table)
    assert recognize(cfg, "1 + (2 + 3)", table, recognizer="ll1")
    assert recognize(cfg, "1 + (2 * 3)", table, recognizer="ll1")
    assert recognize(cfg, "1 + ((2 * 3) + 1)", table, recognizer="ll1")
    assert recognize(cfg, "1 + ((2 * 3) + 1) * 1 + 4", table, recognizer="ll1")


def test_ll1_does_not_recognize():
    table, g = GRAMMAR1
    cfg = parse_grammar(g, table)
    with pytest.raises(SyntaxError):
        recognize(cfg, "(1 + (2 + 3) +", table, recognizer="ll1")
    with pytest.raises(SyntaxError):
        recognize(cfg, "(1 + (2 * 3) +", table, recognizer="ll1")
    with pytest.raises(SyntaxError):
        recognize(cfg, "(1 + ((2 ** 3) + 1)", table, recognizer="ll1")
    with pytest.raises(SyntaxError):
        recognize(cfg, "1 + ((2 * 3) + 1)) + ", table, recognizer="ll1")
