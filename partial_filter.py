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
    def fit(self, S: np.ndarray, m: int, k: int):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        """
        self.n = len(S)
        self.m = m
        if k is None:
            self.k = self.optimal_k()
        else:
            self.k = k
        self.hashes = [universal_hashing() for i in range(self.k)]
        self.a = np.zeros(m)
        for item in S:
            for hash in self.hashes:
                self.a[hash(item) % m] += 1


class PartialFilter(FilterInterface):
    def fit(self, S: np.ndarray, m: int, k: Optional[int], alpha: float, method: int=0):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        alpha: portion of elements to actually store
        method: way of removing elements. 1 means heuristic 1; 2 means heuristic 2; 0 means random.
        """
        self.n = round(len(S) * alpha) # TODO: check if this causes problems
        self.m = m
        if k is None:
            self.k = self.optimal_k()
        else:
            self.k = k
        self.hashes = [universal_hashing() for i in range(self.k)]
        self.a = np.zeros(m)

        if method == 0:
            S_new = np.random.choice(S, self.n, replace=False)
            for item in S_new:
                for hash in self.hashes:
                    self.a[hash(item) % m] += 1
        elif method == 1:
            self.heuristic_1(S)
        elif method == 2:
            self.heuristic_2(S)


    def heuristic_1(self, S):
        """
        Remove elements according to heuristic 1, namely to remove those with most load 1 bits.
        """
        for item in S:
            for hash in self.hashes:
                self.a[hash(item) % self.m] += 1

        count_load_one = []
        for i, item in enumerate(S):
            load = len([1 for hash in self.hashes if self.a[hash(item) % self.m] == 1])
            count_load_one.append((i, load))
        count_load_one = np.array(sorted(count_load_one, key=lambda x: x[1]))
        S_new = S[count_load_one[0:self.n]]

        self.a = np.zeros(self.m)
        for item in S_new:
            for hash in self.hashes:
                self.a[hash(item) % self.m] += 1


    def heuristic_2(self, S):
        # TODO: implement this
        pass


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

        return np.argmin(np.abs(LHS - RHS)) / N

    # For graphing
    @staticmethod
    def plot_errors(alpha_range, FPR, FNR, opt=None, title='error plot', filename='error'):
        plt.figure()
        plt.plot(alpha_range, FPR, label='FPR')
        plt.plot(alpha_range, FNR, label='FNR')
        plt.plot(alpha_range, FPR + FNR, label='total error')
        if opt:
            plt.axvline(opt, color='r', label='opt')
        plt.xlabel('proportion stored (alpha)')
        plt.ylabel('error rates')
        plt.legend()
        plt.title(title)
        # plt.savefig(filename)
        plt.show()
    @staticmethod
    def error_rates_uniform(n, m, p, N=100, plot=True):
        """
        Evaluate the output of optimal_beta for given m,n,p.

        --- Input ---
        n: size of S.
        m: length of array.
        p: |S| / U.
        N: number of intervals to discretize [0,1] into.
        plot: whether to plot the graph.

        --- Output ---
        (FPR, FNR): each an np array of length N.
        """
        opt = PartialFilter.optimal_alpha(n, m, p)
        n_prime_range = np.linspace(1, n, N)

        F_prime = np.array([(2 ** (-m / c / n_prime)) for n_prime in n_prime_range])
        FPR = (1 - p) * F_prime
        FNR = np.array([p * (n - n_prime) * (1 - F_prime[i]) / n for i, n_prime in enumerate(n_prime_range)])

        if plot:
            PartialFilter.plot_errors(n_prime_range/n, FPR, FNR, opt=PartialFilter.optimal_alpha(n, m, p))

        return FPR, FNR

    @staticmethod
    def error_rates_zipfian(n, m, p, N=100, plot=True):
        """
        Evaluate the output of optimal_beta for given m,n,p.
        --- Input ---
        n: size of S.
        m: length of array.
        p: |S| / U.
        N: number of intervals to discretize [0,1] into.
        plot: whether to plot the graph.

        --- Output ---
        (FPR, FNR): each an np array of length N.
        """
        n_prime_range = np.linspace(1, n, N)

        F_prime = np.array([(2 ** (-m / c / n_prime)) for n_prime in n_prime_range])
        FPR = (1 - p) * F_prime
        FNR = np.array([(1 - F_prime[i]) * p * (np.log(n) - np.log(n_prime)) / np.log(n)
                        for i, n_prime in enumerate(n_prime_range)])

        if plot:
            PartialFilter.plot_errors(n_prime_range / n, FPR, FNR, opt=PartialFilter.optimal_alpha(n, m, p))

        return FPR, FNR

    @staticmethod
    def FlogsquaredF(n, m, p, N=100, plot=True):
        n_prime_range = np.linspace(1, n, N)
        F_prime_range = [2 ** (-m / c / n_prime) for n_prime in n_prime_range]
        FlogsquaredF_range = [F * (np.log2(F) ** 2) for F in F_prime_range]

        if plot:
            plt.figure()
            plt.plot(n_prime_range / n, F_prime_range, label='F_prime')
            plt.plot(n_prime_range / n, FlogsquaredF_range, label='F * log_2^2 F')
            plt.axhline(m * p / n, color='r', label='mp/n')
            plt.xlabel('proportion stored (alpha)')
            plt.ylabel('error rates')
            plt.legend()
            plt.title('FlogsquaredF, m/n={}, p={}'.format(m / n, p))
            plt.show()

        return FlogsquaredF_range

    @staticmethod
    def FlogminusF(n, m, p, N=100, plot=True):
        n_prime_range = np.linspace(1, n, N)
        F_prime_range = [2 ** (-m / c / n_prime) for n_prime in n_prime_range]
        FlogminusF_range = [F * (-np.log2(F))  for F in F_prime_range]

        if plot:
            plt.figure()
            plt.plot(n_prime_range / n, F_prime_range, label='F_prime')
            plt.plot(n_prime_range / n, FlogminusF_range, label='F * log_2^2 F')
            plt.axhline(m * p / n, color='r', label='mp/n')
            plt.xlabel('proportion stored (alpha)')
            plt.ylabel('error rates')
            plt.legend()
            plt.title('FlogsquaredF, m/n={}, p={}'.format(m / n, p))
            plt.show()

        return FlogminusF_range


class RetouchedFilter(FilterInterface):
    def fit(self, S, m, k, beta):
        """
        --- Input ---
        S: array of integers.
        m: size of the array
        k: number of hash functions. Default is to calculate optimal k.
        beta: portion of bits to reset
        """
        self.n = len(S)
        self.m = m
        if k is None:
            self.k = self.optimal_k()
        else:
            self.k = k
        self.hashes = [universal_hashing() for i in range(self.k)]
        self.a = np.zeros(m)
        for item in S:
            for hash in self.hashes:
                self.a[hash(item) % m] += 1

        # Retouching
        nonzero_bits = [i for i, b in enumerate(self.a) if b != 0]
        n_reset_bits = round(beta * len(nonzero_bits))
        reset_bit_inds = np.random.choice(nonzero_bits, n_reset_bits, replace=False)
        for i in reset_bit_inds:
            self.a[i] = 0



