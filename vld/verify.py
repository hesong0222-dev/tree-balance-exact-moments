"""Verification of the exact mean and variance of the variance of leaf depths
sigma^2_N(T) = (1/n) sum_x (delta_T(x) - mean)^2  under the Yule and uniform models.

Checks
 (1) brute force over ALL (2n-3)!! labelled rooted binary trees, n <= 9
     (uniform = plain average, Yule = weight 2^{n-1}/n! prod_v 1/(n_v-1));
 (2) independent exact recursion for the joint moments E[S^a Q^b] over root splits, n <= NREC;
 (3) the generating-function identities stated in section.tex (Taylor coefficients).
All arithmetic is exact (Fractions).
"""
from fractions import Fraction as Fr
from math import comb, factorial
from collections import Counter
import time

# ------------------------------------------------------------------ closed forms
def H(n, r=1):
    return sum(Fr(1, i ** r) for i in range(1, n + 1))


def rho(n):  # 4^{n-1}/binom(2n-2,n-1) = (2n-2)!!/(2n-3)!!
    return Fr(4 ** (n - 1), comb(2 * n - 2, n - 1))


def EY(n):
    return Fr(2 * (n + 1), n) * H(n) + Fr(1, n) - 5


def VY(n):
    h1, h2 = H(n), H(n, 2)
    return (Fr(124 * n ** 3 + 1095 * n ** 2 + 2336 * n - 27, 9 * n ** 3)
            - Fr(4 * (n * n - 1), n * n) * h2
            - Fr(2 * (13 * n * n + 98 * n + 77), n ** 3) * h1
            - Fr(8 * (n + 1), n * n) * h1 * h1)


def EU(n):
    return Fr((2 * n - 1) * (n - 1), 3 * n) - Fr(n - 1, 2 * n) * rho(n)


def VU(n):
    r = rho(n)
    return (Fr((n - 1) * (2 * n - 1) * (28 * n ** 3 + 408 * n ** 2 + 221 * n + 117), 315 * n ** 3)
            - Fr((n - 1) * (22 * n ** 3 + 69 * n ** 2 + 29 * n - 6), 48 * n ** 3) * r
            - Fr((n - 1) ** 2, 4 * n * n) * r * r)


# ------------------------------------------------------------------ (1) brute force
def insertions(t, k):
    yield (t, k)
    if isinstance(t, tuple):
        for a in insertions(t[0], k):
            yield (a, t[1])
        for b in insertions(t[1], k):
            yield (t[0], b)


def stats(t):
    """returns (#leaves, S, Q, prod_{internal v}(n_v-1)) computed from leaf depths"""
    depths = []
    prod = 1

    def rec(u, d):
        nonlocal prod
        if not isinstance(u, tuple):
            depths.append(d)
            return 1
        k = rec(u[0], d + 1) + rec(u[1], d + 1)
        prod *= (k - 1)
        return k
    nn = rec(t, 0)
    return nn, sum(depths), sum(d * d for d in depths), prod


def brute(NMAX):
    level = [1]
    out = {}
    for nn in range(1, NMAX + 1):
        if nn > 1:
            if nn < NMAX:
                level = [s for t in level for s in insertions(t, nn)]
                trees = level
            else:
                trees = (s for t in level for s in insertions(t, nn))
        else:
            trees = level
        cnt = Counter()
        for t in trees:
            k, S, Q, prod = stats(t)
            cnt[(k * Q - S * S, prod)] += 1          # D = n^2 sigma^2
        N = sum(cnt.values())
        assert N == (1 if nn == 1 else factorial(2 * nn - 2) // (2 ** (nn - 1) * factorial(nn - 1)))
        u1 = Fr(sum(D * c for (D, p), c in cnt.items()), N * nn ** 2)
        u2 = Fr(sum(D * D * c for (D, p), c in cnt.items()), N * nn ** 4)
        cy = Fr(2 ** (nn - 1), factorial(nn))
        tot = sum(Fr(c, p) for (D, p), c in cnt.items()) * cy
        assert tot == 1
        y1 = sum(Fr(D * c, p) for (D, p), c in cnt.items()) * cy / nn ** 2
        y2 = sum(Fr(D * D * c, p) for (D, p), c in cnt.items()) * cy / nn ** 4
        out[nn] = (N, u1, u2 - u1 * u1, y1, y2 - y1 * y1)
    return out


# ------------------------------------------------------------------ (2) moment recursion
NEEDED = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (4, 0)]


def poly_expand(a, b):
    """coefficients of (S_L+S_R+n_L+n_R)^a (Q_L+2S_L+n_L+Q_R+2S_R+n_R)^b,
    keys ((aL,bL,cL),(aR,bR,cR)) = exponents of S,Q,n on each side (pure python)."""
    from itertools import product
    # variables order: SL,QL,NL,SR,QR,NR
    first = [(1, (1, 0, 0, 0, 0, 0)), (1, (0, 0, 0, 1, 0, 0)), (1, (0, 0, 1, 0, 0, 0)), (1, (0, 0, 0, 0, 0, 1))]
    second = [(1, (0, 1, 0, 0, 0, 0)), (2, (1, 0, 0, 0, 0, 0)), (1, (0, 0, 1, 0, 0, 0)),
              (1, (0, 0, 0, 0, 1, 0)), (2, (0, 0, 0, 1, 0, 0)), (1, (0, 0, 0, 0, 0, 1))]
    res = Counter()
    for choice in product(*([first] * a + [second] * b)):
        c = 1
        e = [0] * 6
        for cc, ee in choice:
            c *= cc
            e = [x + y for x, y in zip(e, ee)]
        res[((e[0], e[1], e[2]), (e[3], e[4], e[5]))] += c
    return res


def cat(k):
    return comb(2 * k, k) // (k + 1)


def moment_rec(model, N):
    ex = {ab: poly_expand(*ab) for ab in NEEDED}
    E = {1: {ab: Fr(int(ab == (0, 0))) for ab in NEEDED}}
    for nn in range(2, N + 1):
        E[nn] = {}
        for ab in NEEDED:
            tot = Fr(0)
            for i in range(1, nn):
                j = nn - i
                p = Fr(1, nn - 1) if model == 'Y' else Fr(cat(i - 1) * cat(j - 1), cat(nn - 1))
                s = 0
                for ((a1, b1, c1), (a2, b2, c2)), c in ex[ab].items():
                    s += c * i ** c1 * j ** c2 * E[i][(a1, b1)] * E[j][(a2, b2)]
                tot += p * s
            E[nn][ab] = tot
    return E


def mean_var_from_moments(E, nn):
    m = E[nn]
    eD = nn * m[(0, 1)] - m[(2, 0)]
    eD2 = nn * nn * m[(0, 2)] - 2 * nn * m[(2, 1)] + m[(4, 0)]
    e1 = eD / nn ** 2
    return e1, eD2 / nn ** 4 - e1 * e1, eD, eD2


# ------------------------------------------------------------------ (3) GF identities
def ser_mul(a, b, N):
    c = [Fr(0)] * (N + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(N + 1 - i):
                c[i + j] += x * b[j]
    return c


def yule_gf_coeffs(N):
    """Taylor coeffs of the claimed Yule GFs (variable z, coefficient of z^{n-1}).
    w=1-z, L=-log w.
      F_D  = 4/w^2 + (4L-4)/w^3
      F_D2 = -140/(3w^2) + (48L^2-124L-104)/w^3 - (144L^2+204L+60)/w^4 + (96L^2-80L+632/3)/w^5"""
    def wp(k):  # w^{-k}
        return [Fr(comb(k + m - 1, m)) for m in range(N + 1)]
    Ls = [Fr(0)] + [Fr(1, m) for m in range(1, N + 1)]
    L2 = ser_mul(Ls, Ls, N)
    one = [Fr(1)] + [Fr(0)] * N

    def comb_(terms):
        tot = [Fr(0)] * (N + 1)
        for k, (c0, c1, c2) in terms.items():
            base = wp(k)
            part = [c0 * x + c1 * y + c2 * z for x, y, z in zip(one, Ls, L2)]
            prod = ser_mul(base, part, N)
            tot = [x + y for x, y in zip(tot, prod)]
        return tot
    FD = comb_({2: (4, 0, 0), 3: (-4, 4, 0)})
    FD2 = comb_({2: (Fr(-140, 3), 0, 0), 3: (-104, -124, 48), 4: (-60, -204, -144), 5: (Fr(632, 3), -80, 96)})
    return FD, FD2


def unif_gf_coeffs(N):
    """claimed uniform GFs (coefficient of z^n), x = sqrt(1-4z), B=(1-x)/2:
      G_D  = 4 z^2 B x^{-5},
      G_D2 = - z^2 B (x^6+3x^5-23x^4-26x^3+115x^2+69x-147) x^{-11}"""
    def xp(k):  # (1-4z)^{k/2}
        out = []
        r = Fr(1)
        h = Fr(k, 2)
        for m in range(N + 1):
            out.append(r * (-4) ** m)
            r *= (h - m) / (m + 1)
        return out
    z2 = [Fr(0), Fr(0), Fr(1)] + [Fr(0)] * (N - 2)
    Bs = [Fr(0)] + [Fr(cat(m - 1)) for m in range(1, N + 1)]
    pref = ser_mul(z2, Bs, N)
    GD = [4 * c for c in ser_mul(pref, xp(-5), N)]
    P = {6: 1, 5: 3, 4: -23, 3: -26, 2: 115, 1: 69, 0: -147}
    poly = [Fr(0)] * (N + 1)
    for e, c in P.items():
        poly = [a + c * b for a, b in zip(poly, xp(e - 11))]
    GD2 = [-c for c in ser_mul(pref, poly, N)]
    return GD, GD2


# ------------------------------------------------------------------ (4) tables printed in section.tex
YULE_TABLE = {  # w = 1-z, L = -log(1-z)
    (1, 0): "2*L/w**2", (0, 1): "(4*L**2+2*L)/w**2",
    (2, 0): "(8*L**2+8*L+6)/w**3-(4*L**2+10*L+6)/w**2",
    (1, 1): "(16*L**3+32*L**2+32*L+14)/w**3-(8*L**3+36*L**2+42*L+14)/w**2",
    (0, 2): "(32*L**4+96*L**3+152*L**2+120*L+70)/w**3-(16*L**4+112*L**3+236*L**2+186*L+70)/w**2",
    (3, 0): "(48*L**3+120*L**2+156*L+72)/w**4-(48*L**3+192*L**2+264*L+150)/w**3+(8*L**3+60*L**2+122*L+78)/w**2",
    (2, 1): "(96*L**4+368*L**3+672*L**2+636*L+252)/w**4-(96*L**4+576*L**3+1240*L**2+1296*L+590)/w**3+(16*L**4+176*L**3+588*L**2+754*L+338)/w**2",
    (4, 0): "(384*L**4+1664*L**3+3456*L**2+3552*L+Rational(4544,3))/w**5-(576*L**4+3360*L**3+7776*L**2+9000*L+4176)/w**4"
            "+(224*L**4+1888*L**3+5576*L**2+7352*L+3978)/w**3-(16*L**4+240*L**3+1132*L**2+2058*L+Rational(3950,3))/w**2",
}
UNIF_TABLE = {  # x = sqrt(1-4z)
    (1, 0): "(x-1-1/x+1/x**2)/4", (0, 1): "(-x+3-1/x-3/x**2+2/x**3)/4",
    (2, 0): "(-x+1+1/x+4/x**2-5/x**3-5/x**4+5/x**5)/8",
    (1, 1): "(x-3-3/x-2/x**2+29/x**3-13/x**4-27/x**5+18/x**6)/8",
    (0, 2): "(-x+9-3/x-20/x**2-65/x**3+163/x**4-7/x**5-152/x**6+76/x**7)/8",
    (3, 0): "(x-1-1/x+7/x**2-63/x**3+27/x**4+177/x**5-123/x**6-114/x**7+90/x**8)/16",
    (2, 1): "(-x+3+3/x+9/x**2+163/x**3-495/x**4-189/x**5+1251/x**6-418/x**7-768/x**8+442/x**9)/16",
    (4, 0): "(-x+1+1/x+16/x**2-407/x**3+1342/x**4+1454/x**5-7344/x**6+1029/x**7+11193/x**8-5391/x**9-5208/x**10+3315/x**11)/32",
}


def check_tables(N, EYm, EUm):
    import sympy as sp
    z = sp.Symbol('z')
    w, L, x = sp.symbols('w L x')
    for ab, s in YULE_TABLE.items():
        e = sp.sympify(s).subs({w: 1 - z, L: -sp.log(1 - z)})
        ser = sp.series(e, z, 0, N).removeO()
        for m in range(N):
            assert sp.Rational(ser.coeff(z, m)) == sp.Rational(EYm[m + 1][ab].numerator, EYm[m + 1][ab].denominator), ('Ytab', ab, m)
    for ab, s in UNIF_TABLE.items():
        e = sp.sympify(s)
        P = sp.Poly(sp.expand(e * x ** 12), x)
        for nn in range(1, N + 1):
            val = Fr(0)
            for (k,), c in P.terms():
                h = Fr(k - 12, 2)
                r = Fr(1)
                for i in range(nn):
                    r *= (h - i) / (i + 1)
                val += Fr(int(sp.numer(c)), int(sp.denom(c))) * r * (-4) ** nn
            assert val == EUm[nn][ab] * cat(nn - 1), ('Utab', ab, nn)
    # Lemma (Yule coefficients) for k=1..5, j=0,1,2
    for k in range(1, 6):
        e0 = (1 - z) ** (-k)
        for j in range(3):
            ser = sp.series(e0 * (-sp.log(1 - z)) ** j, z, 0, N).removeO()
            for m in range(N):
                NN = m + k - 1
                a = H(NN) - H(k - 1)
                b2 = H(NN, 2) - H(k - 1, 2)
                f = [Fr(1), a, a * a - b2][j] * comb(NN, k - 1)
                assert sp.Rational(ser.coeff(z, m)) == sp.Rational(f.numerator, f.denominator), ('lemma', k, j, m)


if __name__ == '__main__':
    t0 = time.time()
    NB = 9
    bf = brute(NB)
    for nn in range(1, NB + 1):
        N, u1, uv, y1, yv = bf[nn]
        ok = (u1 == EU(nn) and uv == VU(nn) and y1 == EY(nn) and yv == VY(nn))
        print(f"brute n={nn} trees={N}: E_U={u1} Var_U={uv} E_Y={y1} Var_Y={yv}  {'OK' if ok else 'FAIL'}")
        assert ok
    print(f"brute force done ({time.time() - t0:.1f}s)")

    NREC = 60
    EYm, EUm = moment_rec('Y', NREC), moment_rec('U', NREC)
    FD, FD2 = yule_gf_coeffs(NREC)
    GD, GD2 = unif_gf_coeffs(NREC)
    for nn in range(1, NREC + 1):
        e1, v, eD, eD2 = mean_var_from_moments(EYm, nn)
        assert e1 == EY(nn) and v == VY(nn), ('Yule', nn)
        assert FD[nn - 1] == eD and FD2[nn - 1] == eD2, ('Yule GF', nn)
        e1, v, eD, eD2 = mean_var_from_moments(EUm, nn)
        assert e1 == EU(nn) and v == VU(nn), ('Unif', nn)
        assert GD[nn] == eD * cat(nn - 1) and GD2[nn] == eD2 * cat(nn - 1), ('Unif GF', nn)
    print(f"OK: closed forms = moment recursion (Yule and uniform) for 1 <= n <= {NREC}")
    print(f"OK: generating functions F_D, F_D2 (Yule) and G_D, G_D2 (uniform) match for 1 <= n <= {NREC}")
    check_tables(16, EYm, EUm)
    print("OK: tables M_ab (Yule) and G_ab (uniform) of section.tex and Lemma (Yule coefficients) checked for n <= 16")
    print("sample values n=10,20,60:")
    for nn in (10, 20, 60):
        print(f"  n={nn}: Var_Y={float(VY(nn)):.6f}  Var_U={float(VU(nn)):.6f}  Var_U/n^2={float(VU(nn)) / nn ** 2:.6f}")
    print(f"total time {time.time() - t0:.1f}s")
