from utils.grammars import GRAMMAR3
from grammar import Grammar
from recognizers import recognize


def test_cyk_recognizer():
    cfg = Grammar.from_str(GRAMMAR3)
    assert recognize(cfg, "book the flight through Houston", recognizer='cyk')
