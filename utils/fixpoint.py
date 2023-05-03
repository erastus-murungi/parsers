def fixpoint(f, max_iterations=1000):
    def helper(arg):
        iterations = 0
        while iterations < max_iterations:
            result = f(arg)
            if result == arg:
                return arg
            arg = result
            iterations += 1
        raise RuntimeError(f"Too many iterations for function {f}")

    return helper


_initial_missing = object()


def reduce(function, sequence, initial=_initial_missing):
    it = iter(sequence)

    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value"
            ) from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value
