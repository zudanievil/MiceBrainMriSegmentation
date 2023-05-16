import os
import random
import numpy as np
from BrainS.lib.math import sum_mean_std, Iterator

# copy from tests/trees
FIX_RANDOM_SEED = not os.getenv(
    "RANDOM_TESTS"
)  # disable randomness unless RANDOM_TESTS in environment
RANDOM_SEED = 42  # applies only when FIX_RANDOM_SEED
RANDOM_TEST_REPETITIONS = 100


def set_seed(seed=RANDOM_SEED) -> "rng state":
    state = random.getstate()
    if FIX_RANDOM_SEED:
        random.seed(seed)
    return state


def rand_loop(n_iter: int = RANDOM_TEST_REPETITIONS) -> Iterator[int]:
    for seed in range(n_iter, RANDOM_SEED + n_iter):
        set_seed(seed)
        yield seed


# end of copy


def test_reduce_mean_std():
    RTOL = 1.0e-5  # relative float tolerance
    for _ in rand_loop():
        samples = np.random.random_sample((10, 10))
        v_mean = samples.mean(axis=0)
        v_std = samples.std(axis=0)
        v_count = np.full(samples.shape[1], samples.shape[0])

        r_mean, r_std, r_count = sum_mean_std(v_mean, v_std, v_count)
        assert np.allclose(r_count, samples.size, rtol=RTOL)
        assert np.allclose(r_mean, samples.mean(), rtol=RTOL)
        assert np.allclose(r_std, samples.std(), rtol=RTOL)
