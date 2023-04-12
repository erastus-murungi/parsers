from recognizers.recognizers import LL1Recognizer
from utils import Tokenizer
from utils.grammars import GRAMMAR1
from utils.parse_grammar import parse_grammar


def test_recognizers_calc_grammar():
    table, g = GRAMMAR1
    cfg = parse_grammar(g, table)
    tks = Tokenizer("1 + (2 + 3)", table).get_tokens_no_whitespace()
    assert LL1Recognizer(cfg).recognizes(tks)

    tks = Tokenizer("1 + (2 * 3)", table).get_tokens_no_whitespace()
    assert LL1Recognizer(cfg).recognizes(tks)
