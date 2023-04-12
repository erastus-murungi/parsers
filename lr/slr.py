from lr.core import Reduce
from lr.lr0 import LR0ParsingTable


class SLRParsingTable(LR0ParsingTable):
    def compute_reduce_actions(self):
        follow_set = self.grammar.follow()
        for state in self.states:
            for item in state.yield_finished():
                for symbol in follow_set[item.name]:
                    if (state, symbol.id) not in self:
                        self[(state, symbol.id)] = Reduce(item.name, len(item.rule))
                    else:
                        raise ValueError(
                            f"Encountered shift/reduce conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.id}\n"
                            f"  {self[(state, symbol.id)]} and \n"
                            f"  Reduce({item.name!s} -> {item.rule!s})"
                        )
