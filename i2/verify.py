"""Equal-weights Colless index I2: exact verification of mean/variance formulas
under the Yule and uniform (PDA) models.

I2(T) = 1/(n-2) * sum_{v internal, n_v>2} |n_L(v)-n_R(v)| / (n_v-2)   (n>=3),  I2 = 0 for n<=2.
Write S(T) = (n-2) I2(T)  (additive, root toll t(i,j)=|i-j|/(i+j-2), t(1,1)=0).
"""
from fractions import Fraction as Fr
from math import comb, factorial, log, pi
from collections import Counter
import time

t0 = time.time()

def b(m): return comb(2 * m, m) if m >= 0 else 0
def Cat(m): return comb(2 * m, m) // (m + 1)
def H(n): return sum(Fr(1, i) for i in range(1, n + 1))
def Aalt(N): return sum(Fr((-1) ** (i + 1), i) for i in range(1, N + 1))
def toll(i, j): return Fr(abs(i - j), i + j - 2) if i + j >= 3 else Fr(0)

# ---------------- formulas ----------------
def muY(n):                      # E_Y[S_n], Theorem (Yule mean)
    if n < 3: return Fr(0)
    corr = Fr(n - 3, 6 * (n - 2)) if n % 2 else Fr(5 * n - 4, 6 * (n - 1))
    return Fr(2 * n, 3) * Aalt(n - 1) - corr

_mu = {}
def mu(n):
    if n not in _mu: _mu[n] = muY(n)
    return _mu[n]

def psi(k):                      # Var of conditional mean given root split
    if k < 3: return Fr(0)
    v = [toll(i, k - i) + mu(i) + mu(k - i) for i in range(1, k)]
    return sum(x * x for x in v) / (k - 1) - mu(k) ** 2

def VY_S(n):                     # Var_Y[S_n]
    return psi(n) + 2 * n * sum(psi(k) / (k * (k + 1)) for k in range(3, n))

def beta(k): return b((k - 1) // 2) * b(k // 2)

_T = {}
def Ttot(n):                     # total of S over the C_{n-1} plane trees with n leaves
    if n not in _T:
        _T[n] = sum(b(n - k) * Fr(b(k - 1) - beta(k), k - 2) for k in range(3, n + 1))
    return _T[n]

def Ttot_harm(n):                # same, with the b-part summed in closed form
    if n < 3: return Fr(0)
    return (b(n - 1) - 2 * b(n - 2) + 4 * b(n - 2) * (H(2 * n - 4) - H(n - 2))
            - sum(b(n - k) * Fr(beta(k), k - 2) for k in range(3, n + 1)))

def F2tot(n):                    # total of S^2 over plane trees
    s = Fr(0)
    for k in range(3, n + 1):
        Q = Fr(k * k * Cat(k - 1) - 4 ** (k - 1), (k - 2) ** 2)
        P = sum(Ttot(i) * Ttot(k - i) for i in range(1, k))
        Hk = sum(Fr(abs(2 * i - k), k - 2) * Ttot(i) * Cat(k - i - 1) for i in range(1, k))
        s += b(n - k) * (Q + 2 * P + 4 * Hk)
    return s

def EY(n): return mu(n) / (n - 2) if n >= 3 else Fr(0)
def VY(n): return VY_S(n) / (n - 2) ** 2 if n >= 3 else Fr(0)
def EU(n): return Fr(Ttot(n), Cat(n - 1)) / (n - 2) if n >= 3 else Fr(0)
def VU(n):
    if n < 3: return Fr(0)
    m1 = Fr(Ttot(n), Cat(n - 1)); m2 = Fr(F2tot(n), Cat(n - 1))
    return (m2 - m1 * m1) / (n - 2) ** 2

# ---------------- brute force over all labelled trees ----------------
def labelled_trees(n):
    """All rooted binary phylogenetic trees on leaves 0..n-1 (leaf insertion)."""
    def insert(t, leaf):           # all trees obtained by attaching `leaf` on an edge of t
        yield (t, leaf)            # edge above the root of t
        if isinstance(t, tuple):
            l, r = t
            for x in insert(l, leaf): yield (x, r)
            for x in insert(r, leaf): yield (l, x)
    def gen(m):
        if m == 1:
            yield 0; return
        for t in gen(m - 1):
            yield from insert(t, m - 1)
    yield from gen(n)

def stats(t):
    """returns (n, S(t) as Fraction, prod over internal v of (n_v - 1))"""
    if not isinstance(t, tuple): return 1, Fr(0), 1
    n1, s1, p1 = stats(t[0]); n2, s2, p2 = stats(t[1])
    n = n1 + n2
    s = s1 + s2 + (Fr(abs(n1 - n2), n - 2) if n > 2 else 0)
    return n, s, p1 * p2 * (n - 1)

def I2_def(t):
    """I2 directly from the definition (list of internal vertices)."""
    nodes = []
    def walk(x):
        if not isinstance(x, tuple): return 1
        a = walk(x[0]); c = walk(x[1]); nodes.append((a, c)); return a + c
    n = walk(t)
    if n <= 2: return Fr(0)
    return Fr(1, n - 2) * sum(Fr(abs(a - c), a + c - 2) for a, c in nodes if a + c > 2)

print("== brute force over all labelled trees ==")
for n in range(3, 10):
    cnt = Counter(); total = 0
    for t in labelled_trees(n):
        nn, s, p = stats(t)
        if n <= 7: assert s / (n - 2) == I2_def(t)
        cnt[(s, p)] += 1; total += 1
    assert total == factorial(2 * n - 2) // (2 ** (n - 1) * factorial(n - 1))  # (2n-3)!!
    Eu = sum(c * s for (s, p), c in cnt.items()) / total / (n - 2)
    Vu = sum(c * s * s for (s, p), c in cnt.items()) / total / (n - 2) ** 2 - Eu ** 2
    yw = Fr(2 ** (n - 1), factorial(n))
    probs = {k: yw / k[1] for k in cnt}
    assert sum(c * probs[k] for k, c in cnt.items()) == 1
    Ey = sum(c * probs[(s, p)] * s for (s, p), c in cnt.items()) / (n - 2)
    Vy = sum(c * probs[(s, p)] * s * s for (s, p), c in cnt.items()) / (n - 2) ** 2 - Ey ** 2
    ok = (Eu == EU(n) and Vu == VU(n) and Ey == EY(n) and Vy == VY(n)
          and Ttot(n) == Ttot_harm(n))
    assert ok, n
    print(f"n={n}: trees={total}  E_U={Eu}  Var_U={Vu}  E_Y={Ey}  Var_Y={Vy}  OK")
print(f"   ({time.time()-t0:.0f}s)")

# ---------------- independent recursion to larger n ----------------
print("== formulas vs independent split recursion (exact) ==")
NM = 45
def rec(p):
    m1 = [Fr(0)] * (NM + 1); m2 = [Fr(0)] * (NM + 1)
    for n in range(2, NM + 1):
        s1 = s2 = Fr(0)
        for i in range(1, n):
            j = n - i; w = p(n, i); tt = toll(i, j)
            s1 += w * (tt + m1[i] + m1[j])
            s2 += w * (tt * tt + m2[i] + m2[j] + 2 * tt * (m1[i] + m1[j]) + 2 * m1[i] * m1[j])
        m1[n] = s1; m2[n] = s2
    return m1, m2
Y1, Y2 = rec(lambda n, i: Fr(1, n - 1))
U1, U2 = rec(lambda n, i: Fr(Cat(i - 1) * Cat(n - i - 1), Cat(n - 1)))
for n in range(3, NM + 1):
    assert EY(n) == Y1[n] / (n - 2) and VY(n) == (Y2[n] - Y1[n] ** 2) / (n - 2) ** 2, n
    assert EU(n) == U1[n] / (n - 2) and VU(n) == (U2[n] - U1[n] ** 2) / (n - 2) ** 2, n
    assert Ttot(n) == Ttot_harm(n)
print(f"n=3..{NM}: Yule E,Var and uniform E,Var (both forms of the mean)  OK   ({time.time()-t0:.0f}s)")

# ---------------- limits of the means ----------------
print("== limit constants of the means (float check) ==")
Gcat = 0.915965594177219015054603514932384110774
cY = 2 * log(2) / 3; cU = (1 + 2 * Gcat) / pi - 0.25
def floatEY(n):
    A = sum((-1) ** (i + 1) / i for i in range(1, n))
    corr = (n - 3) / (6 * (n - 2)) if n % 2 else (5 * n - 4) / (6 * (n - 1))
    return (2 * n / 3 * A - corr) / (n - 2)
for n in (10 ** 3, 10 ** 5):
    print(f"n={n}: E_Y[I2]={floatEY(n):.10f}  limit 2ln2/3={cY:.10f}  n*(E-c)={n*(floatEY(n)-cY):.5f}")
# uniform: n*(E_U - cU) + log(n)/pi should approach a constant
bn = [1.0]
NN = 4001
for m in range(1, NN + 1): bn.append(bn[-1] * (2 * m - 1) / (2 * m))
g = [0.0] * (NN + 1)
for l in range(3, NN + 1): g[l] = (bn[l - 1] - bn[(l - 1) // 2] * bn[l // 2]) / 4 / (l - 2)
for n in (500, 1000, 2000, 4000):
    Tn = sum(bn[n - l] * g[l] for l in range(3, n + 1))
    e = Tn / (bn[n - 1] / n / 4) / (n - 2)
    print(f"n={n}: E_U[I2]={e:.10f}  limit (1+2G)/pi-1/4={cU:.10f}  n*(E-c)+log(n)/pi={n*(e-cU)+log(n)/pi:.4f}")
# the constant as a series
s = sum(g[l] for l in range(3, NN + 1))
print(f"4*sum_(k<=4001) (b_(k-1)-beta_k)/((k-2)4^k) = {4*s:.6f} (tail ~ +{4*2/(4*pi**0.5)/NN**0.5:.4f}) vs cU={cU:.6f}")
# high-precision check of the constant lemma
import mpmath as mp
mp.mp.dps = 30
c = lambda m: (mp.gamma(m + 0.5) / mp.gamma(m + 1)) ** 2 / mp.pi          # b_m^2/16^m
S1 = mp.nsum(lambda j: c(j) / j, [1, mp.inf])
X = mp.nsum(lambda m: c(m) / (m - 1), [2, mp.inf])
bp = mp.nsum(lambda k: mp.gamma(k - 0.5) / mp.gamma(k) / mp.sqrt(mp.pi) / 4 / (k - 2), [3, mp.inf], method='e')  # b_(k-1)/((k-2)4^k)
K_ = mp.catalan
print("S1 - (4ln2-8G/pi) =", mp.nstr(S1 - (4 * mp.log(2) - 8 * K_ / mp.pi), 3))
print("X  - (1/2-1/pi-2G/pi+ln2) =", mp.nstr(X - (mp.mpf(1) / 2 - 1 / mp.pi - 2 * K_ / mp.pi + mp.log(2)), 3))
print("b-part - (1+2ln2)/8 =", mp.nstr(bp - (1 + 2 * mp.log(2)) / 8, 3))
print("4*(b-part - (1/4)(1/4+X)) - cU =", mp.nstr(4 * (bp - (mp.mpf(1) / 4 + X) / 4) - ((1 + 2 * K_) / mp.pi - mp.mpf(1) / 4), 3))
print(f"total time {time.time()-t0:.0f}s")
