from rich import print as print_rich
from rich.pretty import pretty_repr
from rich.traceback import install

from parse_grammar import parse_grammar
from tokenizer import Tokenizer

install(show_locals=True)


if __name__ == "__main__":

    # g = """
    #         <E>
    #         <E> ::= integer
    #         <E> ::= (<E> <Op> <E>)
    #         <Op> ::= + | *
    # """
    #
    # cfg = ContextFreeGrammar.from_string(g)
    # rprint(pretty_repr(cfg))
    # rprint(pretty_repr(cfg.non_terminals))
    #
    # tokens = Tokenizer("((10 + 7) * 7)").get_tokens_no_whitespace()
    # rprint(pretty_repr(cfg.leftmost_top_down_parsing_dfs(tokens)))
    # rprint(pretty_repr(cfg.nullable()))
    # rprint(pretty_repr(cfg.first()))
    # rprint(pretty_repr(cfg.follow()))
    # rprint(pretty_repr(cfg.parsing_table()))
    # rprint(pretty_repr(cfg.match(tokens)))

    # g = """
    #         <A>
    #         <A> ::= <A>b
    #         <A> ::= c
    # """
    # g = """
    #         <A>
    #         <A> ::= c<B>
    #         <B> ::= <>
    #         <B> ::= b<B>
    # """
    tk_table = {
        "+": "+",
        "(": "(",
        "*": "*",
        ")": ")",
        "/": "/",
        "-": "-",
    }
    g = """
            <E>
            <E> ::= <T><E'>
            <E'> ::= + <T><E'> | <>
            <T> ::= <F><T'>
            <T'> ::= * <F><T'> | <>
            <F> ::= (<E>) | integer | float
    """
    # g = """
    #     <EXPRESSION>
    #     <EXPRESSION> ::= <VALUE>
    #     <EXPRESSION> ::= (<EXPRESSION> <OP> <EXPRESSION>)
    #     <OP> ::= + | * | - | /
    #     <VALUE> ::= integer | float
    # """

    # g = """
    # <Expr>
    # <Expr> ::= <Expr> + <Term> | <Term>
    # <Term> ::= <Term> * <Factor> | <Factor>
    # <Factor> ::= (<Expr>) | i
    # """

    cfg = parse_grammar(g, tk_table)
    print_rich(pretty_repr(cfg))
    # rprint(pretty_repr(cfg.non_terminals))

    tks = Tokenizer("(10 + 10)", tk_table).get_tokens_no_whitespace()
    print_rich(pretty_repr(cfg.leftmost_top_down_parsing_dfs(tks)))
    print_rich(pretty_repr(cfg.nullable()))
    print_rich(pretty_repr(cfg.first()))
    print_rich(pretty_repr(cfg.follow()))
    table = cfg.build_ll1_parsing_table()
    print_rich(str(table))
    print_rich(pretty_repr(cfg.match(tks)))
