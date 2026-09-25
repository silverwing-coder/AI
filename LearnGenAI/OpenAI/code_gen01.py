import math
import time
from statistics import mean
from decimal import Decimal


def estimate_time_complexity(fn, input_generator, sizes, trials=3, warmup=True):
    """Estimate the time complexity of a callable by empirical timing.

    Args:
        fn: A callable that accepts one argument (the generated input).
        input_generator: A callable that accepts a size n and returns an input for fn.
        sizes: Iterable of integer sizes to test, e.g. [100, 200, 400, 800].
        trials: Number of timing trials per input size.
        warmup: Run one warmup call before measuring to reduce startup noise.

    Returns:
        A dict with measured times and the best matching complexity class.
    """
    if warmup and sizes:
        fn(input_generator(sizes[0]))

    measured = []
    for n in sizes:
        args = input_generator(n)
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            fn(args)
            end = time.perf_counter()
            times.append(end - start)
        measured.append((n, mean(times)))

    # Candidate growth functions and how to normalize by them.
    candidates = {
        'O(1)': lambda n: 1,
        'O(log n)': lambda n: math.log(n + 1, 2),
        'O(n)': lambda n: n,
        'O(n log n)': lambda n: n * math.log(n + 1, 2),
        'O(n^2)': lambda n: n * n,
        'O(n^3)': lambda n: n ** 3,
        'O(2^n)': lambda n: 2 ** n,
    }

    def score(candidate):
        values = [Decimal(meantime) / Decimal(candidate(n)) for n, meantime in measured]
        avg = mean(values)
        variance = mean((v - avg) ** 2 for v in values)
        return variance, avg

    scores = {name: score(func) for name, func in candidates.items()}
    best = min(scores.items(), key=lambda item: item[1][0])
    best_name, (best_variance, best_ratio) = best

    return {
        'measured': measured,
        'scores': {name: {'variance': var, 'ratio': ratio} for name, (var, ratio) in scores.items()},
        'best_guess': best_name,
        'best_ratio': best_ratio,
    }


def example_usage():
    def sample_function(data):
        # Example: simple linear scan.
        total = 0
        for value in data:
            total += value
        return total

    def make_list(n):
        return list(range(n))

    # result = estimate_time_complexity(sample_function, make_list, [100000, 200000, 400000, 800000, 1600000, 3200000])
    result = estimate_time_complexity(sample_function, make_list, [100000, 200000, 300000, 400000, 500000, 600000])
    print('Best guess:', result['best_guess'])
    print('Measured:', result['measured'])


if __name__ == '__main__':
    example_usage()
