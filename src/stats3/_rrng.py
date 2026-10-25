"""R's default random number stream: Mersenne-Twister seeded as set.seed().

MASS's lqs subsampling and quantreg's bootstrap draw from R's generator, so
reproducing ``set.seed(seed)`` makes their results reproducible here too.
"""

from __future__ import annotations

import math

import numpy as np


class RRandom:
    """``set.seed(seed)`` with R's defaults (Mersenne-Twister, Inversion, Rejection)."""

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50): # RNG_Init's initial scrambling
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        self._bits = np.random.MT19937()
        # i_seed[0] is the position, which FixupSeeds resets to 624.
        self._bits.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}

    def unif_rand(self, size: int | None = None):
        """R's unif_rand(): genrand_int32 * 2^-32, kept inside (0, 1)."""
        raw = self._bits.random_raw(1 if size is None else size)
        u = raw.astype(float) * 2.3283064365386963e-10
        u = np.where(u <= 0, .5 * 2.328306437080797e-10, np.where(1 - u <= 0, 1 - .5 * 2.328306437080797e-10, u))
        return float(u[0]) if size is None else u

    def unif_index(self, n: int) -> int:
        """R_unif_index: rejection sampling of 0..n-1 from 16-bit chunks."""
        if n <= 0:
            return 0
        bits = math.ceil(math.log2(n))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):
                v = 65536 * v + math.floor(self.unif_rand() * 65536)
            v &= (1 << bits) - 1
            if v < n:
                return v

    def sample_replace(self, n: int, size: int) -> np.ndarray:
        """``sample(n, size, replace=TRUE)``, as 1-based indices."""
        return np.array([self.unif_index(n) + 1 for _ in range(size)], dtype=np.int64)
