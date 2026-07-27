"""Deterministic stand-ins for randomness (spec §3.1).

The package carries a zero-randomness guarantee, but several algorithms still
need what randomness is normally used for: spreading mutation steps over a
search space, resampling for a bootstrap CI, picking experiment points. Those
uses do not actually need unpredictability — they need *even coverage* and
*reproducibility*, which is exactly what a low-discrepancy sequence gives and
an RNG does not.

Two tools, for two different jobs:

* :func:`halton` — the i-th point of a Halton sequence. Successive points fill
  the unit cube evenly by construction, so N mutation steps explore the genome
  better than N uniform draws would, and run number 2 explores it identically.
* :func:`hash_unit` / :func:`hash_index` — a value in ``[0, 1)`` (or an index)
  derived from arbitrary seed material via blake2b. For choices that must be
  stable for a given input rather than evenly spread: which bucket an id falls
  in, which of k branches a tick takes.

Neither is a cryptographic primitive and neither should be used as one.
"""
from __future__ import annotations

import hashlib

# The first primes, used as Halton bases. One base per dimension; bases must be
# pairwise coprime, and small primes give the best low-dimensional coverage.
PRIMES: tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
)

_MAX_DIMENSIONS = len(PRIMES)


def van_der_corput(index: int, base: int) -> float:
    """The ``index``-th term of the van der Corput sequence in ``base``.

    Radical inverse: write the index in ``base`` and mirror the digits around
    the point. Index 0 yields 0.0, which is a corner of the cube rather than a
    typical point — callers that want interior points start at 1.
    """
    if base < 2:
        raise ValueError(f"van der Corput base must be >= 2, got {base}")
    n = int(index)
    if n < 0:
        raise ValueError(f"van der Corput index must be >= 0, got {index}")
    result = 0.0
    fraction = 1.0
    while n > 0:
        fraction /= base
        result += (n % base) * fraction
        n //= base
    return result


def halton(index: int, dimensions: int = 1) -> list[float]:
    """The ``index``-th Halton point in ``dimensions`` dimensions.

    Each coordinate lies in ``[0, 1)``. Consecutive indices give points that
    spread out rather than cluster, which is the whole reason this exists.
    """
    if dimensions < 1:
        return []
    if dimensions > _MAX_DIMENSIONS:
        raise ValueError(
            f"halton supports up to {_MAX_DIMENSIONS} dimensions "
            f"(one prime base each), asked for {dimensions}"
        )
    return [van_der_corput(index, PRIMES[d]) for d in range(dimensions)]


def halton_sequence(count: int, dimensions: int = 1,
                    start: int = 1) -> list[list[float]]:
    """``count`` consecutive Halton points, starting at index ``start``.

    Starting at 1 by default: index 0 is the origin, which as a mutation step
    means "no change" and would waste a slot in every generation.
    """
    return [halton(start + i, dimensions) for i in range(max(0, int(count)))]


def scaled(point: float, low: float, high: float) -> float:
    """Map a unit-interval coordinate onto ``[low, high]``."""
    if high < low:
        low, high = high, low
    return low + (high - low) * max(0.0, min(1.0, float(point)))


def signed(point: float) -> float:
    """Map ``[0, 1)`` onto ``[-1, 1)`` — a step with a direction."""
    return 2.0 * max(0.0, min(1.0, float(point))) - 1.0


def _digest(*material) -> int:
    """blake2b of the joined seed material, as an integer."""
    payload = "\x1f".join(str(m) for m in material).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def hash_unit(*material) -> float:
    """A stable value in ``[0, 1)`` derived from the given material.

    The same inputs always give the same number, across processes and restarts.
    """
    return _digest(*material) / 2 ** 64


def hash_index(count: int, *material) -> int:
    """A stable index in ``range(count)`` derived from the given material."""
    if count <= 0:
        raise ValueError(f"hash_index needs a positive count, got {count}")
    return _digest(*material) % int(count)


def hash_choice(options, *material):
    """Pick one of ``options`` stably. Returns None for an empty sequence."""
    items = list(options)
    if not items:
        return None
    return items[hash_index(len(items), *material)]


def bootstrap_indices(n: int, draws: int, replicate: int = 0) -> list[int]:
    """Indices for one deterministic bootstrap resample of ``n`` items.

    A bootstrap needs resampling *with replacement* and only needs the resamples
    to be spread over the index space — which the Halton sequence provides, with
    the bonus that a confidence interval computed twice is the same interval.
    ``replicate`` shifts the sequence so successive replicates differ.
    """
    if n <= 0 or draws <= 0:
        return []
    offset = replicate * draws + 1
    # No clamp is needed: van_der_corput returns a value strictly below 1, so
    # int(v * n) can never reach n. A `min(n - 1, ...)` guard here would be
    # unreachable code that no test could ever distinguish from its absence.
    return [int(van_der_corput(offset + i, 2) * n) for i in range(draws)]
