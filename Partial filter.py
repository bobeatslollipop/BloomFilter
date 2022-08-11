# This is for running experiments on the idea of stroring S' \subseteq S instead of the whole set S.
# Parameters: p, n, U
# Trade-off parameters: m, n', F'(R).
import random
import matplotlib.pyplot as plt

import numpy as np


# Generating hash functions via index.
# See https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python#:~:text=The%20universal%20hash%20family%20is,drawn%20randomly%20from%20set%20H%20.

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
# _memomask = {}
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
        """
        --- variables ---
        n: number of elements actually stored.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        hashes: list of hash functions.
        """
        self.a = self.n = self.m = self.k = self.hashes = None

    def optimal_k(self):
        """
        --- Input ---
        self: has well-defined self.m and self.n values.

        --- Output ---
        k: optimal k value given by int(np.log(2) * self.m / self.n)
        """
        if self.n == 0:
            return 0
        return int(np.log(2) * self.m / self.n)

    def __contains__(self, item: int) -> bool:
        for hash in self.hashes:
            if self.a[hash(item) % self.m] == 0:
                return False
        return True


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
        self.m = m
        if k is None:
            self.k = self.optimal_k()
        else:
            self.k = k

        self.a = np.zeros(m)
        self.hashes = [universal_hashing() for i in range(self.k)]
        for item in S:
            for hash in self.hashes:
                self.a[hash(item) % m] += 1


class PartialFilter(FilterInterface):
    def __init__(self, S: np.ndarray, m: int, k: int | None, alpha: float):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        alpha: portion of elements to actually store
        """
        super().__init__()
        self.n = int(len(S) * alpha)
        self.m = m
        if k is None:
            self.k = self.optimal_k()
        else:
            self.k = k

        S_new = np.random.choice(S, self.n, replace=False)
        self.a = np.zeros(m)
        self.hashes = [universal_hashing() for i in range(self.k)]
        for item in S_new:
            for hash in self.hashes:
                self.a[hash(item) % m] += 1

    @staticmethod
    def calculate_optimal_beta(n, m, p):
        """
        Calculate optimal value of beta for given m,n,p, via the formula
        --- Input ---
        n: size of S.
        m: length of array.
        p: porportion of U that is in S.
        """
        return


class RetouchedFilter(FilterInterface):
    def __init__(self, S, m, k, beta):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        beta: portion of bits to reset
        """
        super().__init__()
        self.n = len(S)
        self.m = m
        if k is None:
            self.k = self.optimal_k()
        else:
            self.k = k

        self.a = np.zeros(m)
        self.hashes = [universal_hashing() for i in range(self.k)]
        for item in S:
            for hash in self.hashes:
                self.a[hash(item) % m] += 1

        # Retouching
        nonzero_bits = [i for i, b in enumerate(self.a) if b != 0]
        n_reset_bits = int(beta * len(nonzero_bits))
        reset_bit_inds = np.random.choice(nonzero_bits, n_reset_bits, replace=False)
        for i in reset_bit_inds:
            self.a[i] = 0


# Basic unit of testing
def avg_errors(n_draws: int, U: int, S: np.ndarray, F: FilterInterface):
    """
    --- Input ---
    n_trials: number of Monte Carlo trials to run.
    n_draws: number of draws in each trial.
    U: universe is [1, U].
    S: the ground truth set.

    --- Output ---
    (FPR, FNR), where error rates are defined as number of errors divided by number of draws
    """
    np.random.seed(None)
    FP_counter = 0
    FN_counter = 0
    draws = np.random.randint(1, U, size=n_draws)
    for draw in draws:
        if draw in F and draw not in S:
            FP_counter += 1
        elif draw not in F and draw in S:
            FN_counter += 1
    FP_rate = FP_counter / n_draws
    FN_rate = FN_counter / n_draws
    return FP_rate, FN_rate


# Testing different choices of alpha for paritial filter.
def test_alpha(alpha_range: np.ndarray,
               m: int,
               n_trials: int,
               n_draws: int,
               U: int,
               S_size: int):
    """
    --- Input ---
    alpha_range: the range of alphas to test.
    m: array length for all filters.
    **other_args: see description for avg_errors.

    --- Output ---
    (FPR, FNR): ndarrays of the corresponding error rates of given alpha.
    """
    N = len(alpha_range)
    FPR = np.zeros((N, n_trials))
    FNR = np.zeros((N, n_trials))
    rng = np.random.default_rng()

    for trial in range(n_trials):
        S = rng.choice(U, size=S_size, replace=False)
        for i, alpha in enumerate(alpha_range):
            F = PartialFilter(S, m, k=None, alpha=alpha)
            FPR[i, trial], FNR[i, trial] = avg_errors(n_draws, U, S, F)

    FPR = np.mean(FPR, axis=1)
    FNR = np.mean(FNR, axis=1)
    # plt.plot(alpha_range, FPR, label='FPR')
    # plt.plot(alpha_range, FNR, label='FNR')
    # plt.plot(alpha_range, FPR + FNR, label='total error')
    # plt.xlabel('stored portion (alpha)')
    # plt.ylabel('error rate')
    # plt.legend()
    # plt.show()

    return FPR, FNR


def test_m(multiplier_range=range(5, 25, 5), n=10000):
    U = 2000000

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('testing different m size')
    ax1.set_xlabel('proportion stored (alpha)')
    ax1.set_ylabel('False Positives')

    ax2.set_xlabel('proportion stored (alpha)')
    ax2.set_ylabel('False Negatives')

    ax3.set_xlabel('proportion stored (alpha)')
    ax3.set_ylabel('Total error')

    for multiplier in multiplier_range:
        alpha_range = np.linspace(0.1, 1, 100)
        FPR, FNR = test_alpha(alpha_range=alpha_range,
                              m=multiplier * n,
                              n_trials=50,
                              n_draws=10000,
                              U=U,
                              S_size=10000)

        ax1.plot(alpha_range, FPR, label='m={}n'.format(multiplier))
        ax2.plot(alpha_range, FNR, label='m={}n'.format(multiplier))
        ax3.plot(alpha_range, FPR + FNR, label='m={}n'.format(multiplier))

    ax3.legend()
    fig.tight_layout()
    plt.savefig('test_m')


test_m()
