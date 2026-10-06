"""Derivation of exact mixed moments E[S^a Q^b] (a+2b... up to degree 4 in depths)
under the Yule and the uniform model, via generating functions.

S = sum of leaf depths, Q = sum of squared leaf depths.
Recursion for T=(L,R), n=n_L+n_R:
  S = S_L+S_R+n,  Q = Q_L+Q_R+2(S_L+S_R)+n.

Yule:    EGF over increasing trees, M(z)=sum_m E[X_{m+1}] z^m, ring Q[w,1/w,L], w=1-z, L=-log w.
Uniform: OGF over plane trees by leaves, ring Q[x,1/x], x=sqrt(1-4z).
Outputs closed forms as sympy expressions (pickled to closed_forms.txt).
"""
from fractions import Fraction as Fr
from itertools import product
from collections import defaultdict
import sympy as sp

# monomials needed
NEEDED = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (4, 0)]


def expand_monomial(a, b):
    """(S_L+S_R+n_L+n_R)^a (Q_L+2S_L+n_L+Q_R+2S_R+n_R)^b as dict
    ((a1,b1,c1),(a2,b2,c2)) -> int coefficient, exponents of S,Q,n on left/right."""
    SL, QL, NL, SR, QR, NR = sp.symbols('SL QL NL SR QR NR')
    e = sp.expand((SL + SR + NL + NR) ** a * (QL + 2 * SL + NL + QR + 2 * SR + NR) ** b)
    out = {}
    for mon, c in sp.Poly(e, SL, QL, NL, SR, QR, NR).terms():
        out[((mon[0], mon[1], mon[2]), (mon[3], mon[4], mon[5]))] = int(c)
    return out


# ---------------------------------------------------------------- Yule ring
class YR:
    """sum c * w^k L^j"""

    def __init__(self, d=None):
        self.d = defaultdict(Fr)
        if d:
            for k, v in d.items():
                if v != 0:
                    self.d[k] += v

    def __add__(s, o):
        r = YR(s.d)
        for k, v in o.d.items():
            r.d[k] += v
        return YR({k: v for k, v in r.d.items() if v != 0})

    def scale(s, c):
        return YR({k: v * c for k, v in s.d.items()})

    def __mul__(s, o):
        r = defaultdict(Fr)
        for (k1, j1), v1 in s.d.items():
            for (k2, j2), v2 in o.d.items():
                r[(k1 + k2, j1 + j2)] += v1 * v2
        return YR(r)

    def deriv(s):  # d/dz
        r = defaultdict(Fr)
        for (k, j), v in s.d.items():
            if k:
                r[(k - 1, j)] += -k * v
            if j:
                r[(k - 1, j - 1)] += j * v
        return YR(r)

    def theta1(s):  # (z d/dz + 1) ; z = 1 - w
        dd = s.deriv()
        zd = dd + (dd * YR({(1, 0): Fr(1)})).scale(-1)
        return zd + s

    def integ0(s):  # int_0^z
        r = defaultdict(Fr)
        for (k, j), v in s.d.items():
            if k == -1:
                r[(0, j + 1)] += v / (j + 1)
            else:
                c = [Fr(0)] * (j + 1)
                c[j] = Fr(-1, k + 1)
                for i in range(j - 1, -1, -1):
                    c[i] = (i + 1) * c[i + 1] / (k + 1)
                for i in range(j + 1):
                    r[(k + 1, i)] += v * c[i]
        const = sum(v for (k, j), v in r.items() if j == 0)
        r[(0, 0)] -= const
        return YR(r)

    def series(s, N):
        """exact Taylor coefficients up to z^N"""
        # w^k: k>=0 polynomial, k<0 binomial series; L = sum z^m/m
        def wpow(k):
            c = [Fr(0)] * (N + 1)
            for m in range(N + 1):
                c[m] = Fr(sp.binomial(-k + m - 1, m)) if k < 0 else Fr((-1) ** m * sp.binomial(k, m))
            return c
        Lser = [Fr(0)] + [Fr(1, m) for m in range(1, N + 1)]

        def mul(a, b):
            c = [Fr(0)] * (N + 1)
            for i, x in enumerate(a):
                if x:
                    for j in range(N + 1 - i):
                        c[i + j] += x * b[j]
            return c
        tot = [Fr(0)] * (N + 1)
        Lp = {0: [Fr(1)] + [Fr(0)] * N}
        for (k, j), v in s.d.items():
            while j not in Lp:
                mx = max(Lp)
                Lp[mx + 1] = mul(Lp[mx], Lser)
            t = mul(wpow(k), Lp[j])
            tot = [a + v * b for a, b in zip(tot, t)]
        return tot


def yule_moments():
    expans = {ab: expand_monomial(*ab) for ab in NEEDED}
    M = {(0, 0): YR({(-1, 0): Fr(1)})}
    theta_cache = {}

    def th(ab, c):
        key = (ab, c)
        if key not in theta_cache:
            f = M[ab]
            for _ in range(c):
                f = f.theta1()
            theta_cache[key] = f
        return theta_cache[key]
    for ab in NEEDED[1:]:
        R = YR()
        for ((a1, b1, c1), (a2, b2, c2)), coef in expans[ab].items():
            if (a1, b1, c1) == (ab[0], ab[1], 0) and (a2, b2, c2) == (0, 0, 0):
                continue
            if (a2, b2, c2) == (ab[0], ab[1], 0) and (a1, b1, c1) == (0, 0, 0):
                continue
            R = R + (th((a1, b1), c1) * th((a2, b2), c2)).scale(coef)
        # M' = 2 M / w + R  ->  M = w^-2 int_0 w^2 R
        M[ab] = YR({(-2, 0): Fr(1)}) * (YR({(2, 0): Fr(1)}) * R).integ0()
    return M


# coefficient extraction [z^m] w^k L^j in terms of n=m+1 and H_n^{(r)}
n = sp.Symbol('n', positive=True)
Hs = sp.symbols('H1:7')  # H1..H6 = H_n^{(r)}


def Hshift(r, s):
    """H^{(r)}_{n+s} expressed with H_n^{(r)}"""
    if s >= 0:
        return Hs[r - 1] + sum(sp.Integer(1) / (n + i) ** r for i in range(1, s + 1))
    return Hs[r - 1] - sum(sp.Integer(1) / (n - i) ** r for i in range(0, -s))


def bell_derivs(f0, p, j):
    """j-th derivative of f(alpha)=f0*exp(sum_r p_r (alpha-a0)^r /r * ...) given
    log-derivatives: (log f)^{(r)} = (-1)^{r-1}(r-1)! p_r.  Returns f^{(j)}(a0)."""
    # f' = f * g, g = (log f)'; use recursion f^{(i+1)} = sum_k binom(i,k) f^{(k)} g^{(i-k)}
    g = [None] + [(-1) ** (r - 1) * sp.factorial(r - 1) * p[r] for r in range(1, j + 1)]  # g^{(r-1)} = (log f)^{(r)}
    fd = [f0]
    for i in range(j):
        fd.append(sum(sp.binomial(i, k) * fd[k] * g[i - k + 1] for k in range(i + 1)))
    return fd[j]


def coef_wL(k, j):
    """[z^m] w^k L^j, m = n-1, valid for m > max(k,0) (generic n). Uses
    w^{-a} L^j = d^j/da^j w^{-a}, [z^m]w^{-a} = prod_{i=0}^{m-1}(a+i)/m!."""
    m = n - 1
    a = -k
    if a >= 1:  # prod_{i=0}^{m-1}(a+i)/m! = (m+a-1)!/((a-1)! m!), logderiv sums_{i} 1/(a+i)^r
        f0 = sp.factorial(m + a - 1) / (sp.factorial(a - 1) * sp.factorial(m))
        f0 = sp.simplify(sp.combsimp(f0))
        p = [None] + [Hshift(r, a - 2) - sum(sp.Integer(1) / i ** r for i in range(1, a)) for r in range(1, j + 1)]
        # sum_{i=0}^{m-1} 1/(a+i)^r = H^{(r)}_{m+a-1} - H^{(r)}_{a-1} ; m+a-1 = n+a-2
        return sp.expand(bell_derivs(f0, p, j)) if j else f0
    # a = -q <= 0 : f(alpha) = (alpha+q) g(alpha), f^{(j)} = j g^{(j-1)}
    q = -a
    if j == 0:
        return sp.Integer(0)  # polynomial, zero for m>q
    # g(-q) = (-1)^q q! (m-1-q)!/m!
    g0 = (-1) ** q * sp.factorial(q) * sp.Integer(1) / sp.prod([(m - i) for i in range(0, q + 1)])
    # sum_{i != q} 1/(i-q)^r = H^{(r)}_{m-1-q} + (-1)^r H^{(r)}_q
    p = [None] + [Hshift(r, -q - 2 + 0) + (-1) ** r * sum(sp.Integer(1) / i ** r for i in range(1, q + 1)) for r in range(1, j)]
    # m-1-q = n-2-q
    return j * (bell_derivs(g0, p, j - 1) if j > 1 else g0)


def yule_closed(F):
    return sp.together(sum(sp.Rational(v.numerator, v.denominator) * coef_wL(k, j) for (k, j), v in F.d.items()))


# ---------------------------------------------------------------- uniform
X = sp.Symbol('x')
Zx = (1 - X ** 2) / 4
Bx = (1 - X) / 2


def u_theta(f):  # z d/dz, with d/dz = -(2/x) d/dx
    return sp.expand(sp.cancel(Zx * (-2 / X) * sp.diff(f, X)))


def uniform_moments():
    expans = {ab: expand_monomial(*ab) for ab in NEEDED}
    M = {(0, 0): Bx}
    cache = {}

    def th(ab, c):
        if (ab, c) not in cache:
            f = M[ab]
            for _ in range(c):
                f = u_theta(f)
            cache[(ab, c)] = f
        return cache[(ab, c)]
    for ab in NEEDED[1:]:
        R = 0
        for ((a1, b1, c1), (a2, b2, c2)), coef in expans[ab].items():
            if (a1, b1, c1) == (ab[0], ab[1], 0) and (a2, b2, c2) == (0, 0, 0):
                continue
            if (a2, b2, c2) == (ab[0], ab[1], 0) and (a1, b1, c1) == (0, 0, 0):
                continue
            R += coef * th((a1, b1), c1) * th((a2, b2), c2)
        M[ab] = sp.expand(sp.cancel(sp.expand(R) / X))  # M = R/(1-2B)
    return M


bn = sp.Symbol('b')  # b_{n-1}=binom(2n-2,n-1) placeholder ; rho = 4^{n-1}/b_{n-1}
rho = sp.Symbol('rho')


def coef_x(k):
    """[z^n] x^k / C_{n-1}, generic n (n > k/2), in terms of n and rho=4^{n-1}/b_{n-1}.
    C_{n-1} = b_{n-1}/n."""
    if k % 2 == 0 and k >= 0:
        return sp.Integer(0)
    if k % 2 == 0:  # (1-4z)^{-p}: 4^n binom(n+p-1,p-1)
        p = -k // 2
        c = 4 * sp.binomial(n + p - 1, p - 1)
        return sp.expand_func(c) * n * rho
    # (1-4z)^{k/2}: 4^n * binom(n - k/2 - 1, n) = 4^n Gamma(n-k/2)/(Gamma(-k/2) n!)
    # ratio to b_{n-1}= 4^{n-1} Gamma(n-1/2)/(sqrt(pi)(n-1)!)
    h = sp.Rational(k, 2)
    ratio = 4 * sp.gamma(n - h) / (sp.gamma(-h) * sp.factorial(n)) * sp.sqrt(sp.pi) * sp.factorial(n - 1) / sp.gamma(n - sp.Rational(1, 2))
    ratio = sp.simplify(sp.gammasimp(ratio))
    g0 = sp.gamma(n - sp.Rational(1, 2))

    def gfix(e):  # gamma(n+a)/gamma(n-1/2) as rational function of n, a half-integer
        a = sp.simplify(e.args[0] - n) + sp.Rational(1, 2)
        a = int(a)
        if a >= 0:
            return g0 * sp.prod([n - sp.Rational(1, 2) + i for i in range(a)])
        return g0 / sp.prod([n - sp.Rational(1, 2) - i for i in range(1, -a + 1)])
    ratio = sp.cancel(ratio.replace(lambda e: isinstance(e, sp.gamma), gfix))
    return sp.factor(ratio * n)


def uniform_closed(F):
    P = sp.Poly(sp.expand(F * X ** 40), X)
    tot = 0
    for (e,), c in P.terms():
        tot += c * coef_x(e - 40)
    return sp.together(sp.expand(tot))


if __name__ == '__main__':
    import pickle, time
    t = time.time()
    MY = yule_moments()
    print('yule GFs done', time.time() - t)
    MU = uniform_moments()
    print('uniform GFs done', time.time() - t)
    for ab in NEEDED:
        print('Y', ab, 'max w power', min(k for k, j in MY[ab].d), max(k for k, j in MY[ab].d), 'max L', max(j for k, j in MY[ab].d))
        print('U', ab, sp.Poly(sp.expand(MU[ab] * X ** 40), X).degree() - 40, 'lowest x power',
              min(m[0] for m in sp.Poly(sp.expand(MU[ab] * X ** 40), X).monoms()) - 40)
    pickle.dump({'Y': {ab: dict(MY[ab].d) for ab in NEEDED}, 'U': {ab: sp.srepr(MU[ab]) for ab in NEEDED}},
                open('gfs.pkl', 'wb'))
