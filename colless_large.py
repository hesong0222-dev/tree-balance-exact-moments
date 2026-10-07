"""Colless index under the uniform (PDA) model: extended checks.
(1) Theorems mean and var (exact formulas) against an independent root-split recursion, n <= 60, exact rationals.
(2) Theorem asym (six-term expansion) at n = 10^6 and 2*10^6, 40-digit arithmetic, and the value quoted in the
    remark after the theorem.  Usage: python3 colless_large.py"""
from fractions import Fraction as Fr
from math import comb
import mpmath as mp

NMAX = 60
b = [comb(2 * m, m) for m in range(2 * NMAX + 2)]
Cat = [b[m] // (m + 1) for m in range(2 * NMAX + 2)]
def C(m): return Fr(-1, 2) if m == -1 else Cat[m]

# (1a) independent recursion over the root split: P(split i | n) = C_{i-1} C_{n-i-1} / C_{n-1}
E1 = [Fr(0)] * (NMAX + 1); E2 = [Fr(0)] * (NMAX + 1)
for n in range(2, NMAX + 1):
    s1 = s2 = Fr(0)
    for i in range(1, n):
        j = n - i; p = Fr(Cat[i - 1] * Cat[j - 1], Cat[n - 1]); t = abs(i - j)
        s1 += p * (t + E1[i] + E1[j])
        s2 += p * (t * t + E2[i] + E2[j] + 2 * t * (E1[i] + E1[j]) + 2 * E1[i] * E1[j])
    E1[n], E2[n] = s1, s2

# (1b) the paper's formulas
def Z(N): return sum(b[j // 2] * b[(j + 1) // 2] * b[N - j] for j in range(N + 1))
T = [0] * (NMAX + 1)                       # total of Colless over plane trees with k leaves
for k in range(1, NMAX + 1): T[k] = 4 ** (k - 1) - Z(k - 1)
for n in range(1, NMAX + 1):
    mean = Fr(T[n], Cat[n - 1])
    assert mean == E1[n], ("mean", n)
    if n >= 2:
        tot = 0
        for k in range(2, n + 1):
            inner = k * k * Cat[k - 1] - 4 ** (k - 1)
            inner += 2 * sum(T[i] * T[k - i] for i in range(1, k))
            inner += 4 * sum(abs(2 * i - k) * T[i] * Cat[k - i - 1] for i in range(1, k))
            tot += b[n - k] * inner
        assert Fr(tot, Cat[n - 1]) == E2[n], ("second moment", n)
print("OK Thm mean and Thm var agree with the root-split recursion for n <=", NMAX)

# (2) asymptotics: E = (N+1)(1 - sum_j r_{floor(j/2)} r_{ceil(j/2)} r_{N-j}) / r_N,  r_k = b_k / 4^k, N = n-1
mp.mp.dps = 40
def mean_big(n):
    N = n - 1; r = [mp.mpf(1)] * (N + 1)
    for k in range(N): r[k + 1] = r[k] * (2 * k + 1) / (2 * k + 2)
    s = mp.fsum(r[j // 2] * r[(j + 1) // 2] * r[N - j] for j in range(N + 1))
    return (N + 1) * (1 - s) / r[N]
g, L2, pi = mp.euler, mp.log(2), mp.pi
for n in (10**6, 2 * 10**6):
    E = mean_big(n); x = mp.mpf(n)
    three = mp.sqrt(pi) * x**1.5 - 2 / pi * x * mp.log(x) - 2 / pi * (g + 5 * L2 - 1) * x
    tail = mp.log(x) / (2 * pi) + (g + 5 * L2) / (2 * pi)
    q_without = (E - three) / mp.sqrt(x); q_with = (E - three - tail) / mp.sqrt(x)
    print(f"n={n}: (E - first three terms)/sqrt(n) = {mp.nstr(q_without, 7)}; "
          f"with the last two terms removed too = {mp.nstr(q_with, 8)}; -3 sqrt(pi)/8 = {mp.nstr(-3 * mp.sqrt(pi) / 8, 8)}")
    if n == 10**6:
        assert abs(q_with - mp.mpf("-0.6646704")) < 5e-7 and abs(q_without - mp.mpf("-0.66183")) < 5e-5
print("OK remark values reproduced")
