from itertools import islice

from rich.traceback import install

from grammar import Grammar
from parsers.parser import EarleyParser, LL1Parser
from utils.dot import draw_tree
from utils.grammars import (
    GRAMMAR3,
    GRAMMAR_0N1N,
    GRAMMAR_DECAF,
    GRAMMAR_REGEX,
    RESERVED_DECAF,
)

install(show_locals=False)

if __name__ == "__main__":
    from rich import print as rprint
    from rich.pretty import pretty_repr

    cfg = Grammar.from_str(GRAMMAR_DECAF, RESERVED_DECAF)
    rprint(pretty_repr(cfg))

    prog = """
        import java;

        void main (int a, int b, float c) {
        int a, b, i;
        a = 10;
        b = 20;
        if (a > b) {
            a = b;
        } else {
            b = a;
        }
        for (int i = 0; i < 10; i++) {
            a = a + 1;
        }
        return a + b;
        }

    """

    earley_parser = EarleyParser(cfg, prog)
    for i, tree in enumerate(islice(earley_parser.parse(), 2)):
        draw_tree(tree.collapse(), f"tree_earley_{i}.pdf")
