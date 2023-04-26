from grammar import Grammar
from ll.decidability import compute_k
from utils.grammars import GRAMMAR_JSON, GRAMMAR_LL5


def test_ll5():
    g = Grammar.from_str(GRAMMAR_LL5)
    assert compute_k(g) == 5


def test_json_grammar():
    g = Grammar.from_str(GRAMMAR_JSON, transform_regex_to_right=True)
    assert compute_k(g) == 2
