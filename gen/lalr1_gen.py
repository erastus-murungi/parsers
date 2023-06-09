import os
import subprocess
from collections import defaultdict
from hmac import HMAC

from rich.pretty import pretty_repr

from grammar import Grammar, Terminal
from lr import Accept, Goto, LALR1ParsingTable, Reduce, Shift

OUTPUT_DIR = "_generated"
TEMPLATE_DIR = "templates"
TEMPLATE_FILENAME = "parser_template.py"
GENERATED_PARSER_FILE_NAME = "parser_generated.py"


def gen_parser(grammar: Grammar):
    parsing_table = LALR1ParsingTable(grammar)
    states = [state.id for state in parsing_table.states]
    states.sort()

    simplified_table = {}
    expected_tokens = defaultdict(list)
    for state in sorted(parsing_table.states, key=lambda state: state.id):
        for symbol in grammar.terminals | grammar.non_terminals:
            action = parsing_table.get((state, symbol.name), None)
            if action is not None:
                match action:
                    case Shift(next_state):
                        simplified_table[(state.id, symbol.name)] = (
                            next_state.id << 1 | 0b1
                        )
                    case Goto(next_state):
                        simplified_table[(state.id, symbol.name)] = next_state.id << 1
                    case Reduce(lhs, len_rhs):
                        simplified_table[(state.id, symbol.name)] = (lhs.name, len_rhs)
                    case Accept():
                        simplified_table[(state.id, symbol.name)] = -1
                if isinstance(symbol, Terminal):
                    expected_tokens[state.id].append(symbol.name)

    tokenizer = grammar.tokenizer

    parser_template_file_path = os.path.join(TEMPLATE_DIR, TEMPLATE_FILENAME)
    parser_generated_file_path = os.path.join(OUTPUT_DIR, GENERATED_PARSER_FILE_NAME)

    with open(parser_template_file_path, "r") as template, open(
        parser_generated_file_path, "w"
    ) as generated:
        items = (
            f"{identifier!r} : re.compile({pattern.pattern!r}, re.DOTALL)\n"
            for identifier, pattern in tokenizer.patterns.items()
        )

        generated.write(
            template.read()
            .replace(
                '"%patterns%"',
                f"{{{', '.join(items)}}} ",
            )
            .replace('"%filename%"', repr(tokenizer.get_filename()))
            .replace('"%reserved%"', repr(tokenizer.reserved))
            .replace('"%parsing_table%"', pretty_repr(simplified_table))
            .replace('"%states%"', pretty_repr(states))
            .replace('"%expected_tokens%"', pretty_repr(dict(expected_tokens)))
        )
    try:
        subprocess.run(["black", parser_generated_file_path])
    except FileNotFoundError:
        print("Black not found, skipping formatting")

    with open(parser_generated_file_path, "r") as generated:
        generated_str = generated.read()

    with open(parser_generated_file_path, "w") as generated:
        generated.write(
            generated_str.replace(
                "%id%", HMAC(b"key", generated_str.encode(), "sha256").hexdigest()
            )
        )


def generate_files(grammar_str: str):
    g = Grammar.from_str(grammar_str)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gen_parser(g)


if __name__ == "__main__":
    from utils.grammars import GRAMMAR1

    generate_files(GRAMMAR1)
