"""Paired statistics for weight-edit evaluations.

Every closed-loop comparison in weightsmith runs two policies on the same simulator slots: a *reference*
(usually the unedited policy, ``base``) and a *candidate* (an edit, or the same weights under another
instruction). A slot fixes the scene and the noise seeds of one episode, so the two runs of a slot form a pair,
and each pair has one of four outcomes:

==========  ==============================================
``both``    both policies succeed
``fixed``   only the candidate succeeds
``broke``   only the reference succeeds
``neither`` both fail
==========  ==============================================

Only the ``fixed`` and ``broke`` pairs carry information about which policy is better. The tools here are the
ones the paper uses (Section 3 and Appendix B of https://ali-adeeb.com/weightsmith/paper.pdf):

* :func:`mcnemar_exact`: exact two-sided McNemar test on the fixed and broke pairs.
* :func:`wilson`: Wilson score interval for one success rate.
* :func:`paired_bootstrap`: interval for the paired difference, resampling the four outcome types.
* :func:`mde`, :func:`pairs_needed`, :func:`mcnemar_power`: what a comparison of n pairs can detect.
* :func:`holm`: Holm correction when you test several tasks or edits at once.
* :func:`compare`: all of the above from the per-slot outcomes of two arms, with a plain-language report.
* :func:`read_arms`: per-slot outcomes from ``weightsmith/heldout.py`` results, control-arm runs or a CSV.
* :func:`validate_slots`: catches slot ranges the evaluation pool cannot run (whole groups of 16).

Example::

    from weightsmith import stats
    arms = stats.read_arms("heldout/p1-RubiksCubeThenBananaTask-91008-64/results.json")
    c = stats.compare(arms["base"], arms["cand"], cand_name="p1")
    print(c.report())          # 11/64 -> 27/64, 22 fixed / 6 broke, exact p = 0.0037, ...

The same is available as ``weightsmith stats RESULTS.json``. Everything here runs on a CPU with numpy only.
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

__all__ = [
    "ALPHA", "POWER", "CONFIDENCE", "NOISE_DISCORDANCE", "BOOTSTRAP_DRAWS", "MIN_HELDOUT_PAIRS",
    "Slot", "PairedTable", "Comparison", "ArmFile",
    "mcnemar_exact", "wilson", "paired_bootstrap", "mde", "pairs_needed", "mcnemar_power", "holm",
    "compare", "read_arms", "read_arm_file", "parse_outcome", "normalize_slot", "validate_slots", "format_p",
]

ALPHA = 0.05
"""Two-sided significance level used throughout (the benchmark accepts an edit at exact McNemar p < 0.05)."""
POWER = 0.8
"""Power used for detectable-effect calculations."""
CONFIDENCE = 0.95
"""Coverage of the Wilson and bootstrap intervals."""
NOISE_DISCORDANCE = 0.3
"""Share of paired outcomes that differ between two runs of *identical* weights (paper Appendix B: 28-38%)."""
BOOTSTRAP_DRAWS = 20_000
"""Bootstrap resamples, as in the paper (seed 0)."""
MIN_HELDOUT_PAIRS = 64
"""The benchmark scores a submission on at least this many held-out pairs (BENCHMARK.md)."""

Slot = int | str
_OutcomeArm = Mapping[Any, Any] | Sequence[Any]


def _z(p: float) -> float:
    return NormalDist().inv_cdf(p)


def _count(x: Any, name: str) -> int:
    if isinstance(x, bool) or int(x) != x or x < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {x!r}")
    return int(x)


def _check_prob(x: float, name: str, *, open_interval: bool = True) -> float:
    x = float(x)
    ok = 0.0 < x < 1.0 if open_interval else 0.0 <= x <= 1.0
    if not ok:
        raise ValueError(f"{name} must be in {'(0, 1)' if open_interval else '[0, 1]'}, got {x}")
    return x


# =============================================================================== tests and intervals
def mcnemar_exact(fixed: int, broke: int) -> float:
    """Exact two-sided McNemar p-value for ``fixed`` and ``broke`` discordant pairs.

    Under the null hypothesis (the two policies are equally good) each discordant pair is a fair coin, so the
    number of fixed pairs is Binomial(fixed + broke, 1/2). The p-value is twice the smaller tail, capped at 1.
    Concordant pairs (both succeed, both fail) do not enter. Identical to ``weightsmith/heldout.py``.

    >>> mcnemar_exact(81, 2)  # the paper's held-out result, 27/128 -> 106/128
    7.210946989931019e-22
    """
    fixed, broke = _count(fixed, "fixed"), _count(broke, "broke")
    m = fixed + broke
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(fixed, broke) + 1))
    return min(1.0, 2 * tail / 2**m)


def wilson(successes: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Wilson score interval for a success rate of ``successes`` out of ``n``.

    Unlike the textbook ``p +- 1.96 sqrt(p(1-p)/n)``, it stays inside [0, 1] and keeps its coverage near 0/n
    and n/n, which matters for tasks the policy almost always fails.

    >>> [round(x, 3) for x in wilson(27, 128)]
    [0.149, 0.29]
    """
    k, n = _count(successes, "successes"), _count(n, "n")
    if n == 0:
        raise ValueError("n must be positive")
    if k > n:
        raise ValueError(f"successes ({k}) cannot exceed n ({n})")
    z = _z(0.5 + _check_prob(confidence, "confidence") / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class PairedTable:
    """Counts of the four paired outcome types for one comparison (reference vs candidate)."""

    both: int
    """Pairs where both policies succeed."""
    fixed: int
    """Pairs where only the candidate succeeds."""
    broke: int
    """Pairs where only the reference succeeds."""
    neither: int
    """Pairs where both policies fail."""

    def __post_init__(self) -> None:
        for name in ("both", "fixed", "broke", "neither"):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if self.n == 0:
            raise ValueError("a paired table needs at least one pair")

    @classmethod
    def from_outcomes(cls, ref: Sequence[Any], cand: Sequence[Any]) -> PairedTable:
        """Table from two equally long sequences of per-slot outcomes (truthy = success)."""
        if len(ref) != len(cand):
            raise ValueError(f"the arms hold {len(ref)} and {len(cand)} outcomes; pair them by slot first")
        r = [bool(x) for x in ref]
        c = [bool(x) for x in cand]
        pairs = list(zip(r, c, strict=True))
        return cls(both=sum(a and b for a, b in pairs), fixed=sum((not a) and b for a, b in pairs),
                   broke=sum(a and (not b) for a, b in pairs), neither=sum((not a) and (not b) for a, b in pairs))

    @classmethod
    def from_counts(cls, n: int, ref_success: int, cand_success: int, fixed: int, broke: int) -> PairedTable:
        """Table from the usual summary numbers (n pairs, successes of each arm, fixed and broke pairs)."""
        n, rs, cs = _count(n, "n"), _count(ref_success, "ref_success"), _count(cand_success, "cand_success")
        fixed, broke = _count(fixed, "fixed"), _count(broke, "broke")
        both = rs - broke
        neither = n - both - fixed - broke
        if both < 0 or neither < 0 or both + fixed != cs:
            raise ValueError(f"inconsistent counts: n={n}, ref {rs}, cand {cs}, fixed {fixed}, broke {broke}")
        return cls(both, fixed, broke, neither)

    @property
    def cells(self) -> tuple[int, int, int, int]:
        """``(both, fixed, broke, neither)``."""
        return (self.both, self.fixed, self.broke, self.neither)

    @property
    def n(self) -> int:
        """Number of pairs."""
        return self.both + self.fixed + self.broke + self.neither

    @property
    def ref_success(self) -> int:
        return self.both + self.broke

    @property
    def cand_success(self) -> int:
        return self.both + self.fixed

    @property
    def discordant(self) -> int:
        """Pairs whose outcomes differ (fixed + broke)."""
        return self.fixed + self.broke

    @property
    def discordance(self) -> float:
        """Share of pairs whose outcomes differ."""
        return self.discordant / self.n

    @property
    def difference(self) -> float:
        """Candidate minus reference success rate, as a fraction: ``(fixed - broke) / n``."""
        return (self.fixed - self.broke) / self.n


def paired_bootstrap(table: PairedTable | Sequence[PairedTable], *, draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
                     confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Bootstrap interval for the paired success difference (candidate minus reference, as a fraction).

    Each draw resamples n pairs from the four outcome types with their observed frequencies (a multinomial
    draw, which is the same as resampling pairs with replacement) and records ``(fixed - broke) / n``; the interval
    is the central ``confidence`` share of the draws. With the defaults it reproduces the paper's intervals
    (20,000 draws, ``numpy.random.default_rng(0)``).

    Pass several tables (one per task) for an interval of the pooled difference that resamples within each task,
    as the paper does for one edit evaluated on several tasks.

    >>> lo, hi = paired_bootstrap(PairedTable.from_counts(128, 27, 106, 81, 2))
    >>> round(100 * lo, 1), round(100 * hi, 1)
    (52.3, 70.3)
    """
    tables = [table] if isinstance(table, PairedTable) else list(table)
    if not tables:
        raise ValueError("no tables")
    draws = _count(draws, "draws")
    if draws < 1:
        raise ValueError("draws must be positive")
    lo_pct = round(50 * (1 - _check_prob(confidence, "confidence")), 10)
    rng = np.random.default_rng(seed)
    total = sum(t.n for t in tables)
    diff = np.zeros(draws)
    for t in tables:
        d = rng.multinomial(t.n, np.asarray(t.cells, dtype=float) / t.n, size=draws)
        diff += d[:, 1] - d[:, 2]
    lo, hi = np.percentile(diff / total, [lo_pct, 100 - lo_pct])
    return float(lo), float(hi)


# =============================================================================== design: what can be detected
def mde(n: int, discordance: float = NOISE_DISCORDANCE, *, alpha: float = ALPHA, power: float = POWER,
        exact: bool = False) -> float:
    """Smallest paired difference (fraction of pairs) that a comparison of ``n`` pairs detects with ``power``.

    ``discordance`` is the share of pairs whose outcomes differ. Use about 0.3, the rate at which *identical*
    weights flip paired outcomes in this setup (paper Appendix B), unless you have measured your own.

    By default this is the paper's normal approximation ``(z_{1-alpha/2} + z_power) * sqrt(discordance / n)``,
    which gives about 27, 19, 14 and 10 points at 32, 64, 128 and 256 pairs. With ``exact=True`` it is the
    smallest difference at which the *exact* McNemar test (:func:`mcnemar_power`) reaches ``power``; that is
    larger for small n, and ``nan`` if no difference up to ``discordance`` reaches it.

    >>> [round(100 * mde(n, 0.3)) for n in (32, 64, 128, 256)]
    [27, 19, 14, 10]
    """
    n = _count(n, "n")
    if n == 0:
        raise ValueError("n must be positive")
    q = _check_prob(discordance, "discordance", open_interval=False)
    if q == 0:
        return 0.0
    if not exact:
        return (_z(1 - _check_prob(alpha, "alpha") / 2) + _z(_check_prob(power, "power"))) * math.sqrt(q / n)
    power = _check_prob(power, "power")
    crit = _critical_counts(n, alpha)
    if _power_from_crit(n, q, 0.0, crit) < power:
        return math.nan
    lo, hi = 0.0, q
    for _ in range(40):
        mid = (lo + hi) / 2
        if _power_from_crit(n, (q + mid) / 2, (q - mid) / 2, crit) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def pairs_needed(effect: float, discordance: float = NOISE_DISCORDANCE, *, alpha: float = ALPHA,
                 power: float = POWER) -> int:
    """Pairs needed to detect a paired difference of ``effect`` (fraction) with ``power`` (normal approximation).

    The inverse of :func:`mde`: ``ceil(discordance * ((z_{1-alpha/2} + z_power) / effect) ** 2)``.

    >>> pairs_needed(0.25)   # a 25-point gain, 30% of pairs discordant
    38
    """
    e = float(effect)
    if not 0 < e <= 1:
        raise ValueError(f"effect must be in (0, 1], got {effect}")
    q = _check_prob(discordance, "discordance")
    z = _z(1 - _check_prob(alpha, "alpha") / 2) + _z(_check_prob(power, "power"))
    return math.ceil(q * (z / e) ** 2 - 1e-9)


def _binom_pmf(n: int, p: float) -> np.ndarray:
    """Binomial(n, p) probabilities of 0..n, computed in log space."""
    k = np.arange(n + 1)
    if p <= 0.0 or p >= 1.0:
        out = np.zeros(n + 1)
        out[0 if p <= 0.0 else n] = 1.0
        return out
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1)))])
    return np.exp(log_fact[n] - log_fact[k] - log_fact[n - k] + k * math.log(p) + (n - k) * math.log1p(-p))


def _critical_counts(n: int, alpha: float) -> list[int]:
    """For m = 0..n discordant pairs: the largest k with exact two-sided p(k, m - k) < alpha, or -1 if none."""
    alpha = _check_prob(alpha, "alpha")
    out = []
    for m in range(n + 1):
        k, tail, denom = -1, 0, 2**m
        while k + 1 <= m // 2:
            nxt = tail + math.comb(m, k + 1)
            if 2 * nxt / denom >= alpha:
                break
            tail, k = nxt, k + 1
        out.append(k)
    return out


def _power_from_crit(n: int, fixed_rate: float, broke_rate: float, crit: list[int]) -> float:
    q = fixed_rate + broke_rate
    if q <= 0:
        return 0.0
    pi = fixed_rate / q
    weights = _binom_pmf(n, min(q, 1.0))
    total = 0.0
    for m in range(n + 1):
        k = crit[m]
        if k < 0 or weights[m] < 1e-300:
            continue
        pmf = _binom_pmf(m, pi)
        total += float(weights[m]) * float(pmf[: k + 1].sum() + pmf[m - k:].sum())
    return min(1.0, total)


def mcnemar_power(n: int, fixed_rate: float, broke_rate: float, *, alpha: float = ALPHA) -> float:
    """Exact probability that the exact McNemar test rejects (p < alpha) with ``n`` pairs.

    ``fixed_rate`` and ``broke_rate`` are the true shares of pairs that come out fixed and broke; their sum is the
    discordance and their difference the true paired difference. The number of discordant pairs is
    Binomial(n, fixed_rate + broke_rate) and, given m of them, the fixed ones are Binomial(m, fixed share); the
    power sums the rejection probability over both. Use it to check a design before spending episodes.

    >>> round(mcnemar_power(64, 0.245, 0.055), 2)   # 19-point gain, 30% discordant, 64 pairs
    0.78
    """
    n = _count(n, "n")
    f = _check_prob(fixed_rate, "fixed_rate", open_interval=False)
    b = _check_prob(broke_rate, "broke_rate", open_interval=False)
    if f + b > 1:
        raise ValueError("fixed_rate + broke_rate cannot exceed 1")
    return _power_from_crit(n, f, b, _critical_counts(n, alpha))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order.

    When you test several tasks or edits and accept any with p < alpha, the chance of at least one false
    acceptance grows with the number of tests. Holm's step-down correction controls it: sort the m p-values,
    multiply the i-th smallest (i = 0..m-1) by (m - i), make the sequence non-decreasing and cap it at 1. Include
    every test you ran, also those that came out at p = 1 (for example a task whose editor submitted no edit).

    >>> [round(p, 3) for p in holm([0.0037191659, 0.122, 0.607, 1, 1, 1, 1])]  # the controlled study
    [0.026, 0.732, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    p = [float(x) for x in pvalues]
    if any(not 0 <= x <= 1 for x in p):
        raise ValueError("p-values must lie in [0, 1]")
    m = len(p)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: p[j])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


# =============================================================================== comparing two arms
def normalize_slot(slot: Any) -> Slot:
    """Slot label as an int when it is one (``90000``, ``"90000"`` and ``"s90000"`` all give 90000)."""
    if isinstance(slot, bool):
        raise ValueError(f"not a slot: {slot!r}")
    if isinstance(slot, int):
        return slot
    s = str(slot).strip()
    t = s[1:] if s[:1] in ("s", "S") else s
    return int(t) if t.lstrip("-").isdigit() else s


_TRUE = {"1", "true", "t", "yes", "y", "success", "succeeded", "pass", "passed"}
_FALSE = {"0", "false", "f", "no", "n", "fail", "failed", "failure"}


def parse_outcome(value: Any) -> bool | None:
    """One per-slot outcome as True (success), False (failure) or None (missing: crashed or not run).

    Accepts booleans, 0/1, the strings true/false, yes/no, success/fail (any case), an empty string or None
    for a missing slot, and the per-slot records of ``ctl_run.py``-style files (``{"ok": ..., "success": ...}``;
    ``ok: false`` means the episode crashed).
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        if value.get("ok") is False:
            return None
        return parse_outcome(value.get("success"))
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"outcome must be 0 or 1, got {value!r}")
    s = str(value).strip().lower()
    if s == "":
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"cannot read {value!r} as an outcome (use 1/0, true/false or success/fail)")


def _as_mapping(arm: _OutcomeArm) -> dict[Slot, bool | None]:
    if isinstance(arm, Mapping):
        return {normalize_slot(k): parse_outcome(v) for k, v in arm.items()}
    if isinstance(arm, (str, bytes)):
        raise TypeError("an arm is a mapping slot -> outcome or a sequence of outcomes")
    return {i: parse_outcome(v) for i, v in enumerate(arm)}


def _slot_sort_key(s: Slot) -> tuple[int, int | str]:
    return (0, s) if isinstance(s, int) else (1, str(s))


def _fmt_slots(slots: Sequence[Slot]) -> str:
    if not slots:
        return "none"
    if all(isinstance(s, int) for s in slots):
        lo, hi = min(slots), max(slots)  # type: ignore[type-var]
        if hi - lo + 1 == len(slots):
            return f"{lo}-{hi}"
        return f"{len(slots)} slots in {lo}-{hi}"
    return f"{len(slots)} slots"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _pts(x: float) -> str:
    v = 100 * x
    return "0.0" if abs(v) < 0.05 else f"{v:+.1f}"


def format_p(p: float) -> str:
    """A p-value for reports: two significant digits, scientific below 0.001."""
    if p >= 0.995:
        return "1"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.2g}"
    return f"{p:.1e}"


@dataclass(frozen=True)
class Comparison:
    """Everything about one paired comparison; build it with :func:`compare`."""

    ref_name: str
    cand_name: str
    slots: tuple[Slot, ...]
    """Paired slots, sorted."""
    ref_outcomes: tuple[bool, ...]
    cand_outcomes: tuple[bool, ...]
    table: PairedTable
    ref_ci: tuple[float, float]
    """Wilson interval of the reference success rate."""
    cand_ci: tuple[float, float]
    """Wilson interval of the candidate success rate."""
    difference_ci: tuple[float, float]
    """Paired bootstrap interval of the success difference (fraction)."""
    p_value: float
    """Exact two-sided McNemar p-value."""
    mde: float
    """Smallest difference these many pairs detect with 80% power at the noise-floor discordance."""
    dropped: tuple[Slot, ...] = ()
    """Slots that were not paired: present in one arm only, or missing (crashed) in either."""
    alpha: float = ALPHA
    confidence: float = CONFIDENCE
    noise_discordance: float = NOISE_DISCORDANCE
    draws: int = BOOTSTRAP_DRAWS
    seed: int = 0

    @property
    def n(self) -> int:
        return self.table.n

    @property
    def fixed(self) -> int:
        return self.table.fixed

    @property
    def broke(self) -> int:
        return self.table.broke

    @property
    def ref_success(self) -> int:
        return self.table.ref_success

    @property
    def cand_success(self) -> int:
        return self.table.cand_success

    @property
    def ref_rate(self) -> float:
        return self.table.ref_success / self.n

    @property
    def cand_rate(self) -> float:
        return self.table.cand_success / self.n

    @property
    def difference(self) -> float:
        """Candidate minus reference success rate (fraction)."""
        return self.table.difference

    @property
    def discordance(self) -> float:
        return self.table.discordance

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def verdict(self) -> str:
        """``"better"``, ``"worse"`` or ``"no detectable difference"`` (candidate relative to reference)."""
        if self.significant and self.difference > 0:
            return "better"
        if self.significant and self.difference < 0:
            return "worse"
        return "no detectable difference"

    def halves(self) -> tuple[Comparison, Comparison]:
        """The comparison split into the first and second half of the slot range.

        ``weightsmith/heldout.py`` runs the reference first on the first half of the slots and the candidate first
        on the second half; a real effect should show up in both halves. Slots are split by position in the
        sorted list of all slots seen (paired, or dropped inside the paired range).
        """
        every = sorted(set(self.slots) | set(self.dropped), key=_slot_sort_key)
        if all(isinstance(s, int) for s in every):  # ignore slots outside the paired range (another run's slots)
            lo, hi = min(self.slots), max(self.slots)  # type: ignore[type-var]
            every = [s for s in every if lo <= s <= hi]  # type: ignore[operator]
        first = set(every[: len(every) // 2])
        parts = []
        for keep in (lambda s: s in first, lambda s: s not in first):
            idx = [i for i, s in enumerate(self.slots) if keep(s)]
            if not idx:
                raise ValueError("a half has no paired slots")
            parts.append(compare({self.slots[i]: self.ref_outcomes[i] for i in idx},
                                 {self.slots[i]: self.cand_outcomes[i] for i in idx},
                                 ref_name=self.ref_name, cand_name=self.cand_name, alpha=self.alpha,
                                 confidence=self.confidence, noise_discordance=self.noise_discordance,
                                 draws=self.draws, seed=self.seed))
        return parts[0], parts[1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary (rates and differences as fractions)."""
        return {
            "ref": self.ref_name, "cand": self.cand_name, "n": self.n,
            "slots": _fmt_slots(self.slots), "dropped_slots": [str(s) for s in self.dropped],
            "ref_success": self.ref_success, "cand_success": self.cand_success,
            "ref_rate": self.ref_rate, "cand_rate": self.cand_rate,
            "ref_wilson": list(self.ref_ci), "cand_wilson": list(self.cand_ci),
            "both": self.table.both, "fixed": self.fixed, "broke": self.broke, "neither": self.table.neither,
            "discordance": self.discordance, "difference": self.difference,
            "difference_bootstrap": list(self.difference_ci), "mcnemar_exact_p": self.p_value,
            "mde": self.mde, "noise_discordance": self.noise_discordance, "alpha": self.alpha,
            "confidence": self.confidence, "bootstrap_draws": self.draws, "bootstrap_seed": self.seed,
            "verdict": self.verdict,
        }

    def report(self) -> str:
        """Plain-language, multi-line summary."""
        c, r = self.cand_name, self.ref_name
        conf = f"{round(100 * self.confidence)}%"
        w = max(len(r), len(c), 8)
        lines = [
            f"{r} vs {c}: {self.n} paired slots ({_fmt_slots(self.slots)})",
            f"  {r:<{w}} {self.ref_success:>4}/{self.n} = {_pct(self.ref_rate):>6}   "
            f"Wilson {conf}: {_pct(self.ref_ci[0])} to {_pct(self.ref_ci[1])}",
            f"  {c:<{w}} {self.cand_success:>4}/{self.n} = {_pct(self.cand_rate):>6}   "
            f"Wilson {conf}: {_pct(self.cand_ci[0])} to {_pct(self.cand_ci[1])}",
            f"  {'pairs':<{w}} both {self.table.both}, fixed {self.fixed}, broke {self.broke}, "
            f"neither {self.table.neither}  ({_pct(self.discordance)} of pairs differ)",
            f"  {'change':<{w}} {_pts(self.difference)} points   paired bootstrap {conf}: "
            f"{_pts(self.difference_ci[0])} to {_pts(self.difference_ci[1])}",
            f"  {'McNemar':<{w}} exact p = {format_p(self.p_value)}   ({self.fixed} fixed vs {self.broke} broke)",
            f"  {'detects':<{w}} about {100 * self.mde:.0f} points with {self.n} pairs "
            f"(80% power, alpha {self.alpha:g}, {round(100 * self.noise_discordance)}% of pairs differing)",
        ]
        if self.verdict == "no detectable difference":
            v = (f"no detectable difference at alpha {self.alpha:g}"
                 + ("" if abs(self.difference) >= self.mde else
                    "; the observed change is smaller than what this many pairs can detect"))
        else:
            v = f"{c} {self.verdict} than {r} (p < {self.alpha:g})"
        lines.append(f"  {'verdict':<{w}} {v}")
        if self.n < MIN_HELDOUT_PAIRS:
            lines.append(f"  {'note':<{w}} fewer than {MIN_HELDOUT_PAIRS} pairs: treat this as a screen, "
                         "not evidence (BENCHMARK.md scores on >= 64 held-out pairs)")
        if self.dropped:
            lines.append(f"  {'note':<{w}} left out {len(self.dropped)} slot(s) without an outcome in both arms "
                         "(run in one arm only, or crashed): "
                         + ", ".join(str(s) for s in self.dropped[:8]) + (" ..." if len(self.dropped) > 8 else ""))
        return "\n".join(lines)


def compare(ref: _OutcomeArm, cand: _OutcomeArm, *, ref_name: str = "base", cand_name: str = "cand",
            alpha: float = ALPHA, confidence: float = CONFIDENCE, noise_discordance: float = NOISE_DISCORDANCE,
            draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> Comparison:
    """Compare two arms run on the same slots and return every statistic in one :class:`Comparison`.

    Args:
        ref, cand: per-slot outcomes of the reference (usually the unedited policy) and the candidate. Either
            mappings ``slot -> outcome`` (paired by slot; slots in only one arm, or missing in either, are dropped
            and reported) or two equally long sequences in slot order. Outcomes are read by :func:`parse_outcome`.
        ref_name, cand_name: labels for the report.
        alpha: significance level of the verdict.
        confidence: coverage of the Wilson and bootstrap intervals.
        noise_discordance: discordance used for the detectable-effect line (identical reruns flip about 30%).
        draws, seed: bootstrap settings (the paper: 20,000 draws, seed 0).
    """
    r, c = _as_mapping(ref), _as_mapping(cand)
    if not isinstance(ref, Mapping) and not isinstance(cand, Mapping) and len(r) != len(c):
        raise ValueError(f"the arms hold {len(r)} and {len(c)} outcomes; pass mappings slot -> outcome to pair")
    every = sorted(set(r) | set(c), key=_slot_sort_key)
    slots = [s for s in every if r.get(s) is not None and c.get(s) is not None]
    dropped = tuple(s for s in every if s not in set(slots))
    if not slots:
        raise ValueError("the two arms share no slot with an outcome in both")
    ro = tuple(bool(r[s]) for s in slots)
    co = tuple(bool(c[s]) for s in slots)
    table = PairedTable.from_outcomes(ro, co)
    return Comparison(
        ref_name=ref_name, cand_name=cand_name, slots=tuple(slots), ref_outcomes=ro, cand_outcomes=co, table=table,
        ref_ci=wilson(table.ref_success, table.n, confidence), cand_ci=wilson(table.cand_success, table.n, confidence),
        difference_ci=paired_bootstrap(table, draws=draws, seed=seed, confidence=confidence),
        p_value=mcnemar_exact(table.fixed, table.broke),
        mde=mde(table.n, noise_discordance, alpha=alpha) if noise_discordance > 0 else 0.0,
        dropped=dropped, alpha=alpha, confidence=confidence, noise_discordance=noise_discordance, draws=draws,
        seed=seed)


# =============================================================================== reading result files
@dataclass
class ArmFile:
    """Per-slot outcomes read from one file: ``arms[name][slot] -> True | False | None``."""

    path: Path
    kind: str
    """``heldout`` (``weightsmith/heldout.py`` results.json or partial.json), ``arms`` (``{"per_arm": ...}``
    as written by control-arm runs, or a plain ``{arm: {slot: outcome}}`` JSON) or ``csv``."""
    arms: dict[str, dict[Slot, bool | None]]
    summary: dict[str, Any] | None = None
    """The file's own summary (heldout.py results.json), for cross-checking."""

    def default_pair(self) -> tuple[str, str]:
        """The two arms compared when none are named: base and cand for held-out files, else the first two."""
        names = list(self.arms)
        if self.kind == "heldout":
            return "base", "cand"
        if len(names) != 2:
            raise ValueError(f"{self.path} holds {len(names)} arms ({', '.join(names)}); name the two to compare")
        return names[0], names[1]


def read_arm_file(path: str | Path) -> ArmFile:
    """Read per-slot outcomes from a results file.

    Recognized formats:

    * ``weightsmith/heldout.py`` ``results.json``: ``{"summary": {...}, "per_slot": {slot: [base, cand]}}``,
      arms ``base`` and ``cand``; its ``partial.json``: ``{"base": {slot: bool}, "cand": {slot: bool}}``.
    * control-arm runs: ``{"per_arm": {arm: {slot: {"ok": bool, "success": bool, ...}}}}`` (crashed episodes,
      ``ok: false``, count as missing), or a plain ``{arm: {slot: outcome}}`` object.
    * CSV with a header: a ``slot`` column (or the first column) and one column per arm, outcomes as 1/0,
      true/false or success/fail; an empty cell is a missing slot, and lines starting with ``#`` are comments.
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".csv", ".tsv") or not text.lstrip().startswith(("{", "[")):  # CSV
        return _read_csv(p, text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: expected a JSON object")
    if "per_slot" in data:
        arms: dict[str, dict[Slot, bool | None]] = {"base": {}, "cand": {}}
        for slot, pair in data["per_slot"].items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{p}: per_slot[{slot}] should be [base, cand]")
            arms["base"][normalize_slot(slot)] = parse_outcome(pair[0])
            arms["cand"][normalize_slot(slot)] = parse_outcome(pair[1])
        return ArmFile(p, "heldout", arms, data.get("summary"))
    body = data.get("per_arm", data)
    if not isinstance(body, dict) or not body or not all(isinstance(v, dict) for v in body.values()):
        raise ValueError(f"{p}: not a recognized results file (expected per_slot, per_arm or {{arm: {{slot: ...}}}})")
    arms = {str(name): {normalize_slot(s): parse_outcome(v) for s, v in d.items()} for name, d in body.items()}
    kind = "heldout" if set(arms) == {"base", "cand"} and "per_arm" not in data else "arms"
    return ArmFile(p, kind, arms, None)


def _read_csv(p: Path, text: str) -> ArmFile:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]  # drop comments
    tabs = p.suffix.lower() == ".tsv" or (bool(lines) and "\t" in lines[0])
    rows = [r for r in csv.reader(lines, csv.excel_tab if tabs else csv.excel) if any(x.strip() for x in r)]
    if len(rows) < 2:
        raise ValueError(f"{p}: need a header row and at least one data row")
    header = [h.strip() for h in rows[0]]
    lower = [h.lower() for h in header]
    si = lower.index("slot") if "slot" in lower else 0
    names = [h for i, h in enumerate(header) if i != si]
    if not names or len(set(names)) != len(names) or any(not h for h in names):
        raise ValueError(f"{p}: the header needs a slot column and distinct, named arm columns")
    arms: dict[str, dict[Slot, bool | None]] = {h: {} for h in names}
    for line, row in enumerate(rows[1:], start=2):
        row = row + [""] * (len(header) - len(row))
        slot = normalize_slot(row[si])
        for i, h in enumerate(header):
            if i == si:
                continue
            try:
                arms[h][slot] = parse_outcome(row[i])
            except ValueError as e:
                raise ValueError(f"{p}, line {line}, column {h!r}: {e}") from None
    return ArmFile(p, "csv", arms, None)


def read_arms(path: str | Path) -> dict[str, dict[Slot, bool | None]]:
    """``{arm: {slot: outcome}}`` from a results file; see :func:`read_arm_file` for the formats."""
    return read_arm_file(path).arms


# =============================================================================== slot ranges
def validate_slots(start: int, n: int, *, envs: int = 16, halves: bool = True) -> tuple[range, range]:
    """Check that slots ``start .. start + n - 1`` can be run by the evaluation pool; return the two halves.

    One simulator run steps ``envs`` environments together, and slot ``s`` always lives in group ``s // envs``
    (that is what gives it the same seeds in every evaluation). So a slot range must cover whole groups: ``start``
    a multiple of ``envs`` and ``n`` a multiple of ``envs``. ``weightsmith/heldout.py`` also alternates the run
    order by half, so with ``halves=True`` each half must be whole groups too (``n`` a multiple of ``2 * envs``).
    The controlled study's first held-out launch at slot 91000 stopped for this reason (91000 = 16 x 5687.5);
    91008 works.

    >>> validate_slots(91008, 64)
    (range(91008, 91040), range(91040, 91072))
    """
    start, n, envs = int(start), _count(n, "n"), _count(envs, "envs")
    if envs < 1 or n == 0:
        raise ValueError("n and envs must be positive")
    if start % envs:
        below = start - start % envs
        raise ValueError(f"slot start {start} is not a multiple of {envs}: slots run in whole groups of {envs} "
                         f"(group = slot // {envs}); use {below} or {below + envs}")
    step = 2 * envs if halves else envs
    if n % step:
        what = "both halves must be whole groups" if halves else "slots run in whole groups"
        raise ValueError(f"n = {n} is not a multiple of {step} ({what}); use {max(step, n - n % step)} or "
                         f"{n - n % step + step}")
    half = n // 2
    return range(start, start + half), range(start + half, start + n)
