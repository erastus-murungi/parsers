from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR_LL5


def test_ll5():
    g = Grammar.from_str(GRAMMAR_LL5, transform_regex_to_right=True)
    assert recognize(g, "bbcd", recognizer="llk")
