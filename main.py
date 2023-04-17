from rich.traceback import install

from parsers.parser import EarleyParser, LL1Parser
from utils.dot import draw_tree
from utils.grammars import GRAMMAR3, GRAMMAR_0N1N
from utils.parse_grammar import parse_grammar

install(show_locals=False)


if __name__ == "__main__":
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

    table, g = GRAMMAR3

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
    #
    # cfg = parse_grammar(g, table)
    # earley_parser = EarleyParser(cfg, "^a+(?:ab)[a-z]\\w+c{1,3}$", table)
    # trees = [t for t in earley_parser.parse()]
    # print(len(trees))
    # for i, tree in enumerate(trees[:5]):
    #     draw_tree(tree, f"tree_{i}.pdf")

    # cfg = parse_grammar(g, table)
    # earley_parser = EarleyParser(cfg, "0011", table)
    # trees = [t for t in earley_parser.parse()]
    # print(len(trees))
    # for i, tree in enumerate(trees[:5]):
    #     draw_tree(tree, f"tree_{i}.pdf")

    # ll1_parser = LL1Parser(cfg, "01", table)
    # tree = ll1_parser.parse()
    # draw_tree(tree, f"tree_ll1.pdf")

    cfg = parse_grammar(g, table)
    earley_parser = EarleyParser(cfg, "book the flight through Houston", table)
    trees = [t for t in earley_parser.parse()]
    print(len(trees))
    for i, tree in enumerate(trees[:5]):
        draw_tree(tree, f"tree_{i}.pdf")
