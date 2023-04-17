from prettytable import PrettyTable

from grammar import EMPTY, Expansion, Grammar, NonTerminal


class LL1ParsingTable(dict[tuple[NonTerminal, str], Expansion]):
    def __init__(self, grammar: Grammar):
        super().__init__()
        self.grammar: Grammar = grammar
        self.construct()

    def construct(self):
        FOLLOW = self.grammar.gen_follow()
        for origin, expansions in self.grammar.items():
            for expansion in expansions:
                first = self.grammar.first(expansion)
                for terminal in first:
                    if terminal is not EMPTY:
                        self[(origin, terminal.name)] = expansion
                if EMPTY in first:
                    for terminal in FOLLOW[origin]:
                        self[(origin, terminal.name)] = expansion

    def __getitem__(self, item: tuple[NonTerminal, str]) -> Expansion:
        return super().__getitem__(item)

    def __setitem__(self, key: tuple[NonTerminal, str], rule: Expansion):
        origin, terminal_id = key
        if (origin, terminal_id) in self:
            raise ValueError(
                f"grammar not LL(1); "
                f"we have <{origin}, {terminal_id}> "
                f"mapping to {super().__getitem__((origin, terminal_id))!s}, {rule!s}"
            )
        super().__setitem__((origin, terminal_id), rule)

    def to_pretty_table(self) -> PrettyTable:
        table = PrettyTable()

        table.field_names = ["Non Terminals \\ Terminals"] + [
            terminal.name for terminal in self.grammar.terminals
        ]

        for origin in self.grammar.non_terminals:
            row: list[str] = [str(origin)]
            for terminal in self.grammar.terminals:
                if (origin, terminal.name) in self:
                    row.append(str(self[(origin, terminal.name)]))
                else:
                    row.append("error")
            table.add_row(row)

        return table

    def __str__(self):
        return self.to_pretty_table()
