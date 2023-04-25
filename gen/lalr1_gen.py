import subprocess
from collections import defaultdict
from hmac import HMAC

from rich.pretty import pretty_repr

from grammar import Grammar, Terminal
from lr import Accept, Goto, LALR1ParsingTable, Reduce, Shift

GENERATED_FILE_NAME = "lalr1_generated.py"


def entry(grammar_str: str, table: dict[str, str]):
    grammar = Grammar.from_str(grammar_str, table)
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

    with open("template.py", "r") as f:
        temp = f.read()
        temp = temp.replace('"%parsing_table%"', pretty_repr(simplified_table))
        temp = temp.replace('"%states%"', pretty_repr(states))
        temp = temp.replace('"%tokenizer_table%"', pretty_repr(table))
        temp = temp.replace('"%expected_tokens%"', pretty_repr(dict(expected_tokens)))

        with open(GENERATED_FILE_NAME, "w") as f1:
            f1.write(temp)
    try:
        subprocess.run(["black", GENERATED_FILE_NAME])
    except FileNotFoundError:
        print("Black not found, skipping formatting")

    with open(GENERATED_FILE_NAME, "r") as f:
        temp = f.read()
        temp = temp.replace("%id%", HMAC(b"key", temp.encode(), "sha256").hexdigest())
    with open(GENERATED_FILE_NAME, "w") as f:
        f.write(temp)


if __name__ == "__main__":
    from utils.grammars import GRAMMAR1

    entry(*GRAMMAR1)
