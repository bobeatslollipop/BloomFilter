
def synthetic(s=0.5, n_intervals=20) -> tuple:
    """
    Zipf distribution.
    s: parameter for distribution
    n_intervals: number of intervals, also the length of output arrays.

    Output: (g, h), where g is key distributon and h is non-key distribution.
    """
    g = [1 / (i ** s) for i in range(1, n_intervals+1)]
    h = g.copy().reverse()
    return g, h