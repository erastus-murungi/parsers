from grammar import Grammar
from ll.decidability import compute_k
from utils.grammars import GRAMMAR_LL5


def test_ll5():
    g = Grammar.from_str(*GRAMMAR_LL5)
    assert compute_k(g) == 5
