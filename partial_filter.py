# This is for running experiments on the idea of stroring S' \subseteq S instead of the whole set S.
# Parameters: p, n, U
# Trade-off parameters: m, n', F'(R).

import matplotlib.pyplot as plt
from typing import Optional
from helpers import *


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
        k: optimal k value given by np.log(2) * self.m / self.n
        """
        if self.n == 0:
            return 0
        return round(np.log(2) * self.m / self.n)

    def __contains__(self, item: int) -> bool:
        if len(self.hashes) == 0:
            return False
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
    def __init__(self, S: np.ndarray, m: int, k: Optional[int], alpha: float):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        alpha: portion of elements to actually store
        """
        super().__init__()
        self.n = round(len(S) * alpha)
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
    def optimal_alpha(n, m, p, N=100):
        """
        Calculate optimal value of alpha for given m,n,p, via formula 2 in BF Note 4.

        --- Input ---
        n: size of S.
        m: length of array.
        p: |S| / U.
        N: number of intervals to discretize [0,1] into.

        --- Output ---
        alpha: optimal value of alpha (up to 1/N precision)
        """
        n_prime_choices = np.linspace(1, n, N)
        LHS = np.array([(2 ** (-m / c / n_prime)) / (n_prime ** 2) for n_prime in n_prime_choices])
        RHS = (c ** 2) * p / m / n

        plt.figure()
        plt.plot(n_prime_choices / n, LHS, label='LHS')
        plt.axhline(RHS, color='r', label='RHS')
        plt.legend()
        # plt.show ()

        return np.argmin(np.abs(LHS - RHS)) / N

    # For graphing
    @staticmethod
    def error_rate(n, m, p, N=100):
        """
        Evaluate the output of optimal_beta for given m,n,p. Plots graph.
        --- Input ---
        n: size of S.
        m: length of array.
        p: |S| / U.
        N: number of intervals to discretize [0,1] into.
        """
        opt = PartialFilter.optimal_alpha(n, m, p)
        n_prime_range = np.linspace(1, n, N)

        FPR = np.array([2 ** (-m / c / n_prime) for n_prime in n_prime_range])
        FNR = np.array([p * (n - n_prime) / n for n_prime in n_prime_range])

        plt.figure()
        plt.plot(n_prime_range / n, FPR, label='FPR')
        plt.plot(n_prime_range / n, FNR, label='FNR')
        plt.plot(n_prime_range / n, FPR + FNR, label='total error')
        plt.axvline(opt, color='r', label='opt')
        plt.xlabel('proportion stored (alpha)')
        plt.ylabel('error rates')
        plt.legend()
        plt.title('evaluate_opt, m/n={}, p={}'.format(m/n, p))
        plt.savefig('m={}n, p={}.png'.format(int(m/n), p))
        # plt.show()


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
        n_reset_bits = round(beta * len(nonzero_bits))
        reset_bit_inds = np.random.choice(nonzero_bits, n_reset_bits, replace=False)
        for i in reset_bit_inds:
            self.a[i] = 0



