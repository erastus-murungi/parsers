from itertools import islice

from rich.traceback import install

from parsers.parser import EarleyParser, LL1Parser
from utils.dot import draw_tree
from utils.grammars import DECAF_GRAMMAR, GRAMMAR3, GRAMMAR_0N1N, GRAMMAR_REGEX
from utils.parse_grammar import parse_grammar

install(show_locals=False)

if __name__ == "__main__":
    from rich import print as rprint
    from rich.pretty import pretty_repr

    # table = {
    #     "+": "+",
    #     "*": "*",
    #     "(": "(",
    #     ")": ")",
    # }
    # g = """
    #     <Expr> -> <Expr> + <Term> | <Term>
    #     <Term> -> <Term> * <Factor> | <Factor>
    #     <Factor> -> (<Expr>) | integer
    # """
    # table = {
    #     "(": "(",
    #     ")": ")",
    #     "+": "+",
    #     "*": "*",
    # }
    #
    # g = """
    #      <S>
    #      <S> -> <E>
    #      <E> -> integer
    #      <E> -> (<E> <Op> <E>)
    #      <Op> -> + | *
    # """
    # g = """
    # <G>
    # <G> -> <A>
    # <A> -> <> | <B>
    # <B> -> <A>
    #
    # """

    table, g = GRAMMAR_REGEX
    cfg = parse_grammar(g, table)
    earley_parser = EarleyParser(cfg, "^a+(?:ab)[a-z]\\w+c{1,3}$", table)
    trees = [t for t in earley_parser.parse()]
    print(len(trees))
    for i, tree in enumerate(trees[:5]):
        draw_tree(tree, f"tree_{i}.pdf")

    # cfg = parse_grammar(g, table)
    # earley_parser = EarleyParser(cfg, "0011", table)
    # trees = [t for t in earley_parser.parse()]
    # print(len(trees))
    # for i, tree in enumerate(trees[:5]):
    #     draw_tree(tree, f"tree_{i}.pdf")
    # ll1_parser = LL1Parser(cfg, "01", table)
    # tree = ll1_parser.parse()
    # draw_tree(tree, f"tree_ll1.pdf")
    # cfg = parse_grammar(g, table)
    # earley_parser = EarleyParser(cfg, "book the flight through Houston", table)
    # for i, tree in enumerate(islice(earley_parser.parse(), 5)):
    #     draw_tree(tree, f"tree_earley_{i}.pdf")

    # table, g = DECAF_GRAMMAR
    # cfg = parse_grammar(g, table)
    # rprint(pretty_repr(cfg))
    #
    # prog = """
    #     import java;
    #
    #     void main (int a, int b, float c) {
    #     int a, b, i;
    #     a = 10;
    #     b = 20;
    #     if (a > b) {
    #         a = b;
    #     } else {
    #         b = a;
    #     }
    #     for (int i = 0; i < 10; i++) {
    #         a = a + 1;
    #     }
    #     return a + b;
    #     }
    #
    # """
    #
    # earley_parser = EarleyParser(cfg, prog, table)
    # for i, tree in enumerate(islice(earley_parser.parse(), 2)):
    #     draw_tree(tree.collapse(), f"tree_earley_{i}.pdf")
