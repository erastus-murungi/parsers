from collections import defaultdict

from prettytable import PrettyTable

from grammar import CFG, EMPTY, Expansion, NonTerminal


class LL1ParsingTable(dict[tuple[NonTerminal, str], Expansion]):
    def __init__(self, grammar: CFG):
        super().__init__()
        self.grammar: CFG = grammar
        self.build_ll1_parsing_table()

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def to_pretty_table(self) -> str:
        terminals: list[str] = ["<NT>"] + [
            terminal.name for terminal in self.grammar.terminals
        ]
        pretty_table = PrettyTable()
        pretty_table.field_names = terminals
        hashmap: dict[NonTerminal, dict[str, Expansion | str]] = defaultdict(dict)

        for (non_terminal, terminal_id), pt_entries in self.items():
            hashmap[non_terminal][terminal_id] = pt_entries

        for non_terminal, terminal_id2_rule in hashmap.items():
            row: list[Expansion | str] = [str(non_terminal)]
            for terminal in self.grammar.terminals:
                row.append(
                    terminal_id2_rule[terminal.name]
                    if terminal.name in terminal_id2_rule
                    else "error"
                )
            pretty_table.add_row(row)

        return pretty_table.to_string()

    def __str__(self):
        return self.to_pretty_table()

    def __getitem__(self, item: tuple[NonTerminal, str]) -> Expansion:
        return super().__getitem__(item)

    def __setitem__(self, key: tuple[NonTerminal, str], rule: Expansion):
        non_terminal, terminal_id = key
        if (non_terminal, terminal_id) in self:
            raise ValueError(
                f"grammar not LL(1); "
                f"we have <{non_terminal}, {terminal_id}> "
                f"mapping to {super().__getitem__((non_terminal, terminal_id))!s}, {rule!s}"
            )
        super().__setitem__((non_terminal, terminal_id), rule)

    def build_ll1_parsing_table(self):
        follow_set = self.grammar.gen_follow()
        for non_terminal, definition in self.grammar.items():
            for rule in definition:
                for terminal in self.grammar.first(rule):
                    if terminal is not EMPTY:
                        self[(non_terminal, terminal.name)] = rule
                if EMPTY in self.grammar.first(rule):
                    for terminal in follow_set[non_terminal]:
                        self[(non_terminal, terminal.name)] = rule
