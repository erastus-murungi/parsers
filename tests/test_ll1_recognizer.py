import pytest

from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR1, GRAMMAR2


@pytest.mark.parametrize("grammar", [GRAMMAR1, GRAMMAR2])
def test_ll1_recognize(grammar):
    cfg = Grammar.from_str(grammar)
    assert recognize(cfg, "1 + (2 + 3)", recognizer="ll1")
    assert recognize(cfg, "1 + (2 * 3)", recognizer="ll1")
    assert recognize(cfg, "1 + ((2 * 3) + 1)", recognizer="ll1")
    assert recognize(cfg, "1 + ((2 * 3) + 1) * 1 + 4", recognizer="ll1")


def test_ll1_does_not_recognize():
    cfg = Grammar.from_str(GRAMMAR1)
    with pytest.raises(SyntaxError):
        recognize(cfg, "(1 + (2 + 3) +", recognizer="ll1")
    with pytest.raises(SyntaxError):
        recognize(cfg, "(1 + (2 * 3) +", recognizer="ll1")
    with pytest.raises(SyntaxError):
        recognize(cfg, "(1 + ((2 ** 3) + 1)", recognizer="ll1")
    with pytest.raises(SyntaxError):
        recognize(cfg, "1 + ((2 * 3) + 1)) + ", recognizer="ll1")
