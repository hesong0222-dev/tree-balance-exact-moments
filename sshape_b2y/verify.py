"""Exact verification for (A) the sb-shape statistic s^(T)=sum_v log(n_v-1) and
(B) the Yule variance of B2.  Run: python3 verify.py  (about 1-2 minutes)."""
from fractions import Fraction as F
from math import comb, factorial, log, pi, sqrt
import sys, time

NMAX_BRUTE = int(sys.argv[1]) if len(sys.argv) > 1 else 9

def b(m): return comb(2*m, m)
def C(m): return comb(2*m, m)//(m+1)
def dfact(m):  # double factorial, (-1)!!=1
    r = 1
    while m > 1: r *= m; m -= 2
    return r
def H(n, r=1): return sum(F(1, k**r) for k in range(1, n+1))

# ---------------- formulas: clade-count moments ----------------
# N_k = number of internal vertices with n_v = k (2<=k<=n).  s^ = sum_k N_k * l_k, l_k = log(k-1).
def eY(n, k):
    if k > n or k < 2: return F(0)
    return F(1) if k == n else F(2*n, k*(k+1))
def eU(n, k):
    if k > n or k < 2: return F(0)
    return F(C(k-1)*b(n-k), C(n-1))
def dY(n, j, k):  # expected # ordered pairs of disjoint clades of sizes j,k
    s = j+k
    if s > n: return F(0)
    if s == n: return F(2, n-1)
    return F(4*n*(n-s), j*(j+1)*k*(k+1)) + F(4*n*(s*s+j*k+1), (j+1)*(k+1)*(s-1)*s*(s+1))
def dU(n, j, k):
    s = j+k
    if s > n: return F(0)
    m = n-s+2
    return F(C(j-1)*C(k-1)*(m-1)*b(m-1), C(n-1))
def P(e, d, n, j, k):  # E[N_j N_k]
    if j > k: j, k = k, j
    if k > n: return F(0)
    if j == k: return e(n, k) + d(n, k, k)
    return e(n, k)*e(k, j) + d(n, j, k)
PY = lambda n, j, k: P(eY, dY, n, j, k)
PU = lambda n, j, k: P(eU, dU, n, j, k)

# ---------------- formula: Yule variance of B2 (log base 2) ----------------
def VarY_B2(n): return 2 - F(2*b(n-1), 4**(n-1)) - H(n-1, 2)

# ---------------- brute force over all labelled trees ----------------
def insert(t, x):
    yield (t, x)
    if isinstance(t, tuple):
        a, c = t
        for a2 in insert(a, x): yield (a2, c)
        for c2 in insert(c, x): yield (a, c2)

def stats(t, depth, sizes, depths):
    if not isinstance(t, tuple):
        depths.append(depth); return 1
    m = stats(t[0], depth+1, sizes, depths) + stats(t[1], depth+1, sizes, depths)
    sizes.append(m); return m

def brute(n):
    """returns dict model -> (E[N_k] vector, E[N_jN_k] matrix, E[B2], E[B2^2]) with exact Fractions."""
    trees = [1]
    for x in range(2, n):
        trees = [t2 for t in trees for t2 in insert(t, x)]
    acc = {}
    for model in ('U', 'Y'):
        acc[model] = [F(0), [F(0)]*(n+1), [[F(0)]*(n+1) for _ in range(n+1)], F(0), F(0)]
    # aggregate by shape signature to keep exact arithmetic cheap
    sig = {}
    cnt = 0
    for t in trees:
        for t2 in insert(t, n):
            cnt += 1
            sizes, depths = [], []
            stats(t2, 0, sizes, depths)
            key = (tuple(sorted(sizes)), tuple(sorted(depths)))
            sig[key] = sig.get(key, 0) + 1
    assert cnt == dfact(2*n-3), cnt
    for (sizes, depths), mult in sig.items():
        Nk = [0]*(n+1)
        for m in sizes: Nk[m] += 1
        B2 = sum(F(d, 2**d) for d in depths)
        wY = F(2**(n-1), factorial(n))
        for m in sizes: wY /= (m-1)
        for model, w in (('U', F(mult, dfact(2*n-3))), ('Y', wY*mult)):
            A = acc[model]
            A[0] += w
            for k in range(2, n+1):
                if Nk[k]:
                    A[1][k] += w*Nk[k]
                    for j in range(2, n+1):
                        if Nk[j]: A[2][j][k] += w*Nk[j]*Nk[k]
            A[3] += w*B2; A[4] += w*B2*B2
    return acc, cnt

# ---------------- independent recursion (Markov branching) for larger n ----------------
def recursion(nmax, q):
    """exact E[N_k], E[N_j N_k] for sizes up to nmax, root-split law q(n,i)."""
    E = {1: [F(0)]*(nmax+1)}
    M = {1: [[F(0)]*(nmax+1) for _ in range(nmax+1)]}
    for n in range(2, nmax+1):
        e = [F(0)]*(nmax+1); Mm = [[F(0)]*(nmax+1) for _ in range(nmax+1)]
        for i in range(1, n):
            w = q(n, i); L, R = i, n-i
            for k in range(2, n+1):
                e[k] += w*(E[L][k] + E[R][k])
            for j in range(2, n+1):
                aj = E[L][j] + E[R][j]
                for k in range(2, n+1):
                    Mm[j][k] += w*(M[L][j][k] + M[R][j][k] + E[L][j]*E[R][k] + E[R][j]*E[L][k])
        e[n] += 1
        for j in range(2, n+1):
            for k in range(2, n+1):
                Mm[j][k] += (1 if j == n else 0)*(1 if k == n else 0)
            # cross terms with the root (N_n gets +1 from root)
            Mm[n][j] += e[j] - (1 if j == n else 0)
            Mm[j][n] += e[j] - (1 if j == n else 0)
        Mm[n][n] -= 0
        E[n] = e; M[n] = Mm
    return E, M

qY = lambda n, i: F(1, n-1)
qU = lambda n, i: F(C(i-1)*C(n-i-1), C(n-1))

def B2rec(nmax):
    """exact Yule mean and second moment of B2 via B2(T)=1+(B2(L)+B2(R))/2, uniform split."""
    m1 = {1: F(0)}; m2 = {1: F(0)}
    for n in range(2, nmax+1):
        a = sum(m1[i] + m1[n-i] for i in range(1, n))/(n-1)
        s2 = sum(m2[i] + m2[n-i] + 2*m1[i]*m1[n-i] for i in range(1, n))/(n-1)
        m1[n] = 1 + a/2
        m2[n] = 1 + a + s2/4
    return m1, m2

def main():
    t0 = time.time()
    ok = True
    # ---- brute force
    for n in range(2, NMAX_BRUTE+1):
        acc, cnt = brute(n)
        for model, e, PP in (('U', eU, PU), ('Y', eY, PY)):
            A = acc[model]
            assert A[0] == 1
            g1 = all(A[1][k] == e(n, k) for k in range(2, n+1))
            g2 = all(A[2][j][k] == PP(n, j, k) for j in range(2, n+1) for k in range(2, n+1))
            print(f"n={n} ({cnt} labelled trees) model={model}: E[N_k] {'OK' if g1 else 'FAIL'}, E[N_jN_k] {'OK' if g2 else 'FAIL'}")
            ok &= g1 and g2
        A = acc['Y']
        g = A[3] == H(n-1) and A[4]-A[3]**2 == VarY_B2(n)
        print(f"n={n} Yule B2: mean {A[3]}, var {A[4]-A[3]**2}  formula {VarY_B2(n)}  {'OK' if g else 'FAIL'}")
        ok &= g
    # uniform clade probability (exchangeable form) identity
    g = all(eU(n, k) == F(comb(n, k)*dfact(2*k-3)*dfact(2*n-2*k-1), dfact(2*n-3))
            for n in range(2, 60) for k in range(2, n+1))
    print('uniform E[N_k] = binom(n,k)(2k-3)!!(2n-2k-1)!!/(2n-3)!!  n<60:', 'OK' if g else 'FAIL'); ok &= g
    # ---- independent recursion to larger n
    NR = 22
    for name, q, e, PP in (('Y', qY, eY, PY), ('U', qU, eU, PU)):
        E, M = recursion(NR, q)
        g = all(E[n][k] == e(n, k) and all(M[n][j][k] == PP(n, j, k) for j in range(2, n+1))
                for n in range(2, NR+1) for k in range(2, n+1))
        print(f'recursion check model={name}, n<={NR}:', 'OK' if g else 'FAIL'); ok &= g
    m1, m2 = B2rec(80)
    g = all(m1[n] == H(n-1) and m2[n]-m1[n]**2 == VarY_B2(n) for n in range(1, 81))
    print('B2 Yule recursion check n<=80:', 'OK' if g else 'FAIL'); ok &= g
    # BCS21 eq. (8) single sum agrees
    def alpha(n): return (2 - H(n, 2) - H(n)/n)/2
    def bcs(n):
        s = F(0)
        for k in range(2, n):
            beta = alpha(k) - alpha(k-1)*F(k-1, k)
            s += beta*F(dfact(2*k)*1, dfact(2*k-1))
        return F(dfact(2*n-3), dfact(2*n-2))*s, F(dfact(2*n-1), dfact(2*n))*s
    g = all(bcs(n)[0] == VarY_B2(n) for n in range(2, 40))
    print('agreement with BCS21 recurrence solved as (2n-3)!!/(2n-2)!! * sum_k beta_k (2k)!!/(2k-1)!!, n<40:', 'OK' if g else 'FAIL'); ok &= g
    print('  (BCS21 eq. (8) as printed, prefactor (2n-1)!!/(2n)!!, gives n=4:', bcs(4)[1], 'instead of', VarY_B2(4), '-> index typo)')
    # alpha -> 1 limit of BDM26 Prop. 2.17 (numerical, mpmath)
    try:
        import mpmath as mp
        mp.mp.dps = 60
        def varH(n, a):
            a = mp.mpf(a); be = 1 - 2**(1-a)
            f = lambda k: 1 + 2*(2**(1-a)-1)/k
            g_ = lambda k: (2**(1-a)-1)**2/k*mp.fprod([1+(2**(1-2*a)-1)/i for i in range(1, k)])
            Pf = mp.fprod([f(k) for k in range(1, n)])
            S = 1 + mp.fsum([g_(m)/mp.fprod([f(k) for k in range(1, m+1)]) for m in range(1, n)])
            Mn = mp.fprod([1+(2**(1-a)-1)/k for k in range(1, n)])
            return (Pf*S - Mn**2)/be**2
        g = True
        for n in (4, 7, 15, 30):
            v = (varH(n, 1+mp.mpf('1e-20')) + varH(n, 1-mp.mpf('1e-20')))/2
            exact = VarY_B2(n)
            err = abs(v - mp.mpf(exact.numerator)/exact.denominator)
            g &= err < mp.mpf('1e-15')
        print('BDM26 Prop 2.17 at alpha=1 +- 1e-20 matches closed form (n=4,7,15,30):', 'OK' if g else 'FAIL'); ok &= g
    except ImportError:
        print('mpmath missing; skipped alpha->1 check')
    for n in (100, 10000):
        v = float(VarY_B2(n)) if n <= 2000 else 2 - 2/sqrt(pi*(n-1))*(1-1/(8*(n-1))) - (pi**2/6 - 1/(n-1) + 1/(2*(n-1)**2))
        print(f'  Var_Y(B2_n) - (2 - pi^2/6 - 2/sqrt(pi n) + 1/n), n={n}: {v-(2-pi**2/6-2/sqrt(pi*n)+1/n):.2e}')
    print('first values Var_Y(B2), n=1..10:', [str(VarY_B2(n)) for n in range(1, 11)])
    # ---- numerical values / asymptotics (floats, natural log)
    def sY_mean(n): return sum(float(eY(n, k))*log(k-1) for k in range(3, n+1))
    def sU_mean(n): return sum(float(eU(n, k))*log(k-1) for k in range(3, n+1))
    KY = sum(2*log(m)/((m+1)*(m+2)) for m in range(1, 2_000_000)) + 2*(log(2e6)+1)/2e6
    print('Yule mean, n=10,100,1000: ', [round(sY_mean(n), 6) for n in (10, 100, 1000)])
    for n in (100, 1000):
        print(f'  n={n}: E_Y - (K_Y n - ln n - 2) = {sY_mean(n) - (KY*n - log(n) - 2):.3e}   (K_Y={KY:.8f})')
    print('first exact values (coefficient form) E_Y[s], n=2..6:',
          [' + '.join(f'{eY(n,k)}*l{k}' for k in range(3, n+1)) for n in range(3, 7)])
    print('total', round(time.time()-t0, 1), 's;', 'ALL OK' if ok else 'SOME FAIL')

if __name__ == '__main__':
    main()
