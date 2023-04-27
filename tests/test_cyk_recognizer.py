from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR3


def test_cyk_recognizer():
    cfg = Grammar.from_str(GRAMMAR3)
    assert recognize(cfg, "book the flight through Houston", recognizer="cyk")
