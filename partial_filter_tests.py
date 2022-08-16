import partial_filter
from partial_filter import *

def avg_errors(n_draws: int, U: int, S: np.ndarray, F: FilterInterface, zipfian=False):
    """
    Basic unit of testing any filter.

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
    if zipfian:
        # TODO: implement this
        pass
    else:
        draws = np.random.randint(1, U, size=n_draws)
    for draw in draws:
        if draw in F and draw not in S:
            FP_counter += 1
        elif draw not in F and draw in S:
            FN_counter += 1
    FP_rate = FP_counter / n_draws
    FN_rate = FN_counter / n_draws
    return FP_rate, FN_rate


def test_alpha(alpha_range: np.ndarray,
               m: int,
               n_trials: int,
               n_draws: int,
               U: int,
               S_size: int,
               zipfian: bool=False,
               method: int=0):
    """
    Testing different choices of alpha in partial filters. Other parameters are held constant.

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

    for trial in range(n_trials):
        print('starting trial {} of {}'.format(trial, n_trials))
        S = np.random.choice(U, size=S_size, replace=False)
        for i, alpha in enumerate(alpha_range):
            F = PartialFilter()
            F.fit(S, m, k=None, alpha=alpha, method=method)
            FPR[i, trial], FNR[i, trial] = avg_errors(n_draws, U, S, F, zipfian=zipfian)

    FPR = np.mean(FPR, axis=1)
    FNR = np.mean(FNR, axis=1)

    return FPR, FNR


def test_m(multiplier_range=range(7, 22, 3), n=10000):
    """
    Testing different ranges of m, and run test_alpha for each m. Other parameters held constant.
    """
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
        print('testing multiplier {} of {}'.format(multiplier, multiplier_range))
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

    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.tight_layout()
    plt.savefig('test_m_many')


def information_theoretic_D(n=1000, m=10000, p=0.2, N=500):
    """
    Lower possible distortion via information theory.
    R(D) = max(H(p) - H(D), 0)
    """

    H_p = p * (-np.log2(p)) + (1-p) * (-np.log2(1-p))
    H_D = max(H_p - m / (n / p), 0)

    D_range = np.linspace(0, 0.5, N)
    H_D_range = -np.array([D * np.log2(D) + (1-D) * np.log2(1-D) for D in D_range]) # entropy
    H_D_range[0] = 0.0 # originally nan

    return np.argmin(np.abs(H_D_range - H_D)) / (N * 2)


def info_theoretic_test(multiplier_range=np.arange(0.1, 20, 0.1), n=1000, p=0.1):
    """
    Testing different ranges of m, and run test_alpha for each m. Other parameters held constant.
    """
    U = round(n / p)
    BF_distortion = []
    partial_distortion = []
    minimum_distortion = []

    for multiplier in multiplier_range:
        BF_distortion.append(BloomFilter.error_rates(n=n, m=multiplier*n, optimal=True))
        FPR, FNR = PartialFilter.error_rates_uniform(n=n, m=multiplier*n, p=p, N=500, plot=False, optimal=True)
        partial_distortion.append(np.min(FPR + FNR))
        minimum_distortion.append(information_theoretic_D(n=n, m=multiplier*n, p=p, N=10000))

    # plt.plot(multiplier_range * n, BF_distortion, label='distortion from Bloom filter')
    plt.plot(multiplier_range * n, partial_distortion, label='distortion from partial filter')
    plt.plot(multiplier_range * n, minimum_distortion, label='theoretical minimum distortion')
    plt.xlabel('array size(m)')
    plt.ylabel('distortion')
    plt.title('info theoretic comparison, n={}, p={}'.format(n, p))
    plt.legend()
    plt.show()


def test_heuristic1(n=10000, m=100000, U=2000000, N=100):
    alpha_range = np.linspace(0.1, 1, N)
    FPRcomp, FNRcomp = test_alpha(alpha_range=alpha_range,
                              m=m,
                              n_trials=100,
                              n_draws=10000,
                              U=U,
                              S_size=n,
                              method=0)
    FPR, FNR = test_alpha(alpha_range=alpha_range,
                              m=m,
                              n_trials=100,
                              n_draws=10000,
                              U=U,
                              S_size=n,
                              method=1)
    opt = PartialFilter.optimal_alpha(1000, 10000, 0.005)
    plt.plot(alpha_range, FPR + FNR, label='total error from heuristic1')
    plt.plot(alpha_range, FPRcomp + FNRcomp, label='total error from partial filter')
    plt.axvline(opt, color='r', label='theorecitcal optimal')
    plt.xlabel('stored portion (alpha)')
    plt.ylabel('error rate')
    plt.legend()
    plt.savefig('heuristic1.png')


def rounding_error_heuristic2():
    n, m, p, N = (10000, 100000, 0.05, 500)
    FPR, FNR = PartialFilter.error_rates_zipfian(n=n, m=m, p=p, plot=False, N=N)
    FminuslogF = PartialFilter.FminuslogF(n=n, m=m, p=p, plot=False, N=N)
    plt.figure()
    plt.plot(np.linspace(0, 1, N), FPR+FNR, label='error')
    plt.plot(np.linspace(0, 1, N), FminuslogF, label='F * log_2 F')
    plt.axhline(p / np.log2(n), color='r', label='p / logn')
    plt.axvline(np.argmin(FPR+FNR) / N, color='g', label='opt')
    plt.xlabel('proportion stored (alpha)')
    plt.ylabel('error rates')
    plt.legend()
    plt.title('FminuslogF, m/n={}, p={}'.format(m / n, p))
    plt.show()

test_heuristic1()