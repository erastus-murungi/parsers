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
