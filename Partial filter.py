# This is for running experiments on the idea of stroring S' \subseteq S instead of the whole set S.
# Parameters: p, n, U
# Trade-off parameters: m, n', F'(R).
import random
import matplotlib.pyplot as plt

import numpy as np

# Generating hash functions via index.
# See https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python#:~:text=The%20universal%20hash%20family%20is,drawn%20randomly%20from%20set%20H%20.
_memomask = {}


def universal_hashing():
    def rand_prime():
        while True:
            p = random.randrange(2 ** 32, 2 ** 34, 2)
            if all(p % n != 0 for n in range(3, int((p ** 0.5) + 1), 2)):
                return p
    m = 2 ** 32 - 1
    p = rand_prime()
    a = random.randint(0, p)
    if a % 2 == 0:
        a += 1
    b = random.randint(0, p)
    def h(x):
        return ((a * x + b) % p) % m
    return h

# Doesn't work very well
# def hash_function(n: int) -> Callable:
#     mask = _memomask.get(n)
#     if mask is None:
#         random.seed(n)
#         mask = _memomask[n] = random.getrandbits(32)
#
#     def myhash(x):
#         return hash(x) ^ mask
#
#     return myhash


class FilterInterface():
    def __init__(self):
        self.a = np.zeros(0)
        self.n = 0

    def arraylen(self):
        return self.a.shape[0]

    def setsize(self):
        return self.n

    def optimal_k(self, m, n):
        if n == 0:
            return 0
        return int(np.log(2) * m / n)


# Create a regular BF.
class BloomFilter(FilterInterface):
    def __init__(self, S: np.ndarray, m: int, k: int):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        """
        super().__init__()
        self.n = len(S)
        if k is None:
            k = self.optimal_k(m, self.n)

        self.a = np.zeros(m)
        hash_inds = np.random.randint(1, 1000, size=k, dtype=np.int32)
        self.hashes = [universal_hashing() for i in hash_inds]
        for item in S:
            for hash in self.hashes:
                self.a[hash(item) % m] = 1

    def __contains__(self, item: int) -> bool:
        for hash in self.hashes:
            if self.a[hash(item) % m] == 0:
                return False
        return True


class PartialFilter(FilterInterface):
    def __init__(self, S: np.ndarray, m: int, k: int | None, alpha: float):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        alpha:
        """
        super().__init__()
        self.n = int(len(S) * alpha)
        if k is None:
            k = self.optimal_k(m, self.n)

        S_new = np.random.choice(S, self.n, replace=False)
        self.a = np.zeros(m)
        hash_inds = np.random.randint(1, 1000, size=k, dtype=np.int32)
        self.hashes = [universal_hashing() for i in hash_inds]
        for item in S_new:
            for hash in self.hashes:
                self.a[hash(item) % m] = 1

    def __contains__(self, item: int) -> bool:
        for hash in self.hashes:
            if self.a[hash(item) % m] == 0:
                return False
        return True


def avg_errors(n_trials: int, n_draws: int, U: int, S: np.ndarray, F: FilterInterface):
    """
    --- Input ---
    n_trials: number of Monte Carlo trials to run.
    n_draws: number of draws in each trial.
    U: universe is [1, U].
    S: the ground truth set.

    --- Outputs ---
    (FPR, FNR), where error rates are defined as number of errors divided by number of draws
    """
    np.random.seed(None)
    FP_rates = []
    FN_rates = []
    for _ in range(n_trials):
        FP_counter = 0
        FN_counter = 0
        draws = np.random.randint(1, U, size=n_draws)
        for draw in draws:
            if draw in F and draw not in S:
                FP_counter += 1
            elif draw not in F and draw in S:
                FN_counter += 1
        FP_rates.append(FP_counter / n_draws)
        FN_rates.append(FN_counter / n_draws)
    return np.mean(FP_rates), np.mean(FN_rates)


def test_alpha(alpha_range: np.ndarray,
               m: int,
               n_trials: int,
               n_draws: int,
               U: int,
               S: np.ndarray):
    """
    --- Input ---
    alpha_range: the range of alphas to test.
    m: array length for all filters.
    **other_args: see description for avg_errors.

    --- Output ---
    (FPRs, FNRs): ndarrays of the corresponding error rates of given alpha.
    """
    N = len(alpha_range)
    FPRs = np.zeros(N)
    FNRs = np.zeros(N)

    for i, alpha in enumerate(alpha_range):
        F = PartialFilter(S, m, k=None, alpha=alpha)
        FPRs[i], FNRs[i] = avg_errors(n_trials, n_draws, U, S, F)

    plt.plot(alpha_range, FPRs, label='FPR')
    plt.plot(alpha_range, FNRs, label='FNR')
    plt.plot(alpha_range, FPRs + FNRs, label='total error')
    plt.xlabel('stored portion (alpha)')
    plt.ylabel('error rate')
    plt.legend()
    plt.show()

    return FPRs, FNRs


U = 2000000
S = np.array(range(1, 10001))
m = 100000
test_alpha(alpha_range=np.arange(0.1, 1, 0.01),
           m=m,
           n_trials=20,
           n_draws=10000,
           U=U,
           S=S)
