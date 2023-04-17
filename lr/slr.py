from lr.core import Reduce
from lr.lr0 import LR0ParsingTable


class SLRParsingTable(LR0ParsingTable):
    def compute_reduce_actions(self):
        follow_set = self.grammar.gen_follow()
        for state in self.states:
            for item in state.yield_finished():
                for symbol in follow_set[item.name]:
                    if (state, symbol.name) not in self:
                        self[(state, symbol.name)] = Reduce(
                            item.name, len(item.expansion)
                        )
                    else:
                        raise ValueError(
                            f"Encountered conflict on \n"
                            f" state: {str(state)}\n and symbol: {symbol.name}\n"
                            f"  {self[(state, symbol.name)]} and \n"
                            f"  Reduce({item.name!s} -> {item.expansion!s})"
                        )
