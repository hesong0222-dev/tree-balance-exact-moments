"""Numerical leading constants of Var(S_n)/n, S_n=(n-2)I2_n (float64; uses the exact formulas of section.tex)."""
import numpy as np, math, mpmath as mp
# ---- uniform: Var_U(S_n)/n from the generating-function formula, n <= N
N = 20000
k = np.arange(N + 1, dtype=float)
bn = np.ones(N + 1)
for m in range(1, N + 1): bn[m] = bn[m - 1] * (2 * m - 1) / (2 * m)      # b_m/4^m
Cn = bn / (k + 1)                                                          # C_m/4^m
g = np.zeros(N + 1)
for l in range(3, N + 1): g[l] = (bn[l - 1] - bn[(l - 1) // 2] * bn[l // 2]) / 4 / (l - 2)
T = np.convolve(bn, g)[:N + 1]                                             # T_i/4^i
P = np.convolve(T, T)[:N + 1]
Cm1 = np.zeros(N + 1); Cm1[1:] = Cn[:N] / 4                                # C_{j-1}/4^j
R = np.zeros(N + 1)
for kk in range(3, N + 1):
    i = np.arange(1, kk)
    R[kk] = ((kk * kk * Cn[kk - 1] / 4 - 0.25) / (kk - 2) ** 2
             + 4 * np.dot(np.abs(2 * i - kk) * T[i], Cm1[kk - i]) / (kk - 2) + 2 * P[kk])
F2 = np.convolve(bn, R)[:N + 1]
ns = np.array([1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 20000])
V = np.array([F2[n] / (Cn[n - 1] / 4) - (T[n] / (Cn[n - 1] / 4)) ** 2 for n in ns])
print("uniform Var(S_n)/n:", dict(zip(ns.tolist(), np.round(V / ns, 7).tolist())))
L = np.log(ns)
for cols in ([np.ones_like(L), L ** 2 / ns, L / ns, 1 / ns],
             [np.ones_like(L), L ** 2 / ns, L / ns, 1 / ns, L / ns ** 2, 1 / ns ** 2]):
    M = np.array(cols).T; c = np.linalg.lstsq(M, V / ns, rcond=None)[0]
    print("  fit sigma_U^2 =", round(c[0], 8))
# ---- Yule: sigma_Y^2 = 2 sum_k psi_k/(k(k+1))
K = 40001
A = np.zeros(K + 1)
for m in range(1, K + 1): A[m] = A[m - 1] + (-1) ** (m + 1) / m
mu = np.zeros(K + 1)
for m in range(3, K + 1):
    mu[m] = 2 * m / 3 * A[m - 1] - ((m - 3) / (6 * (m - 2)) if m % 2 else (5 * m - 4) / (6 * (m - 1)))
nu = mu - 2 * math.log(2) / 3 * np.arange(K + 1)
psi = np.zeros(K + 1)
for kk in range(3, K + 1):
    i = np.arange(1, kk)
    psi[kk] = np.var(np.abs(2 * i - kk) / (kk - 2) + nu[i] + nu[kk - i])
ks = np.arange(2000, K + 1); Lk = np.log(ks); s = (-1.0) ** ks
basis = lambda kk, Lk, s: [Lk / kk, 1 / kk, Lk / kk ** 2, 1 / kk ** 2, s / kk, s / kk ** 2, Lk ** 2 / kk ** 2, Lk / kk ** 3, 1 / kk ** 3]
M = np.array(basis(ks, Lk, s)).T
c = np.linalg.lstsq(M, psi[ks] - 1 / 12, rcond=None)[0]
part = math.fsum(psi[kk] / (kk * (kk + 1)) for kk in range(3, K + 1))
kt = np.arange(K + 1, 10 ** 7 + 1, dtype=float)
tail = 1 / (12 * (K + 1)) + math.fsum(np.dot(c, np.array(basis(kt, np.log(kt), (-1.0) ** kt))) / (kt * (kt + 1)))
print("psi_k at k=10^4, 4*10^4:", psi[10000], psi[40000], "(limit 1/12 =", 1 / 12, ")")
print("Yule sigma_Y^2 = 2 sum psi_k/(k(k+1)) =", f"{2 * (part + tail):.11f}", "  (tail part", f"{2 * tail:.3e}", ")")
