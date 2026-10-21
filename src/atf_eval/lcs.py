from __future__ import annotations

from collections.abc import Hashable, Sequence


def lcs_length(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def order_similarity(expected: Sequence[Hashable], observed: Sequence[Hashable]) -> float | None:
    """Normalized LCS ratio. None (N/A) when both sequences are empty."""
    if not expected and not observed:
        return None
    denom = max(len(expected), len(observed))
    return lcs_length(expected, observed) / denom


def align(
    expected: Sequence[Hashable], observed: Sequence[Hashable]
) -> list[tuple[int | None, int | None]]:
    """Position-aware LCS alignment: returns (expected_index, observed_index)
    pairs in position order. A matched element gets both indices; a missing
    expected element gets (i, None); an extra observed element gets (None, j).
    This is what "identifies missing, extra and reordered nodes without
    cascading all later nodes into failures" (METRICS.md) means in practice --
    an LCS-based diff/alignment, not positional zip or a bag-of-words count.
    """
    n, m = len(expected), len(observed)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == observed[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected[i - 1] == observed[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            pairs.append((None, j - 1))
            j -= 1
        else:
            pairs.append((i - 1, None))
            i -= 1
    pairs.reverse()
    return pairs
