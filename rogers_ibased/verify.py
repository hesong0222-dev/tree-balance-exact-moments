"""Rogers J index and I-based indices (Total I, Total I', I value, I' value):
exact moments under the Yule and uniform (PDA) models.
Checks (exact rationals): (1) brute force over all labelled rooted binary trees, n<=9;
(2) independent moment recursions, larger n; (3) symbolic telescoping proof of the
closed form for Var_Y[Sigma I']; (4) numerical asymptotic checks.  Runtime ~2-3 min."""
import sys, time
from fractions import Fraction as Fr
from math import comb, factorial, pi, log, sqrt
from functools import lru_cache
from itertools import combinations
from collections import Counter

NMAX_BRUTE = int(sys.argv[1]) if len(sys.argv) > 1 else 9

def Cat(m): return comb(2*m, m)//(m+1) if m >= 0 else 0
def b(m): return comb(2*m, m) if m >= 0 else 0
@lru_cache(None)
def H(n, r=1): return sum((Fr(1, k**r) for k in range(1, n+1)), Fr(0))

# ---------------- tolls (definitions, survey Sec. 9.11 / 9.13) ----------------
def tJ(i, j): return Fr(int(i != j))
def tI(i, j):
    k = i+j
    if k < 4: return Fr(0)
    c = (k+1)//2                       # ceil(k/2)
    return Fr(max(i, j)-c, (k-1)-c)
def corr(k): return Fr(k-1, k) if k % 2 == 0 else Fr(1)
def tIp(i, j): return tI(i, j)*corr(i+j)
TOLL = {'J': tJ, 'I': tI, 'Ip': tIp}

# ---------------- formulas ----------------
def NY(n, k): return Fr(1) if k == n else Fr(2*n, k*(k+1))       # E#vertices of size k, Yule
def NU(n, k): return Fr(Cat(k-1)*b(n-k), Cat(n-1))                # same, uniform

def EY_J(n): return 2*n*(1-H(n)+H(n//2))-(2 if n % 2 == 0 else 0)
def EU_J(n): return n-1-sum((NU(n, 2*m)*Fr(Cat(m-1)**2, Cat(2*m-1)) for m in range(1, n//2+1)), Fr(0))
def EY_sumI(n): return n*(H(n)-H(n//2))-Fr(5*n, 12)-(n % 2) if n >= 4 else Fr(0)
def EY_sumIp(n): return Fr(n, 4)-Fr(1, 2) if n >= 4 else Fr(0)
def VY_sumIp(n):
    small = {1: 0, 2: 0, 3: 0, 4: Fr(1, 8), 5: Fr(5, 8), 6: Fr(23, 36)}
    if n in small: return Fr(small[n])
    h = n//2
    base = Fr(33359, 15120)*n-Fr(4, 9)*n*(H(n)-H(h))-Fr(13, 12)*n*H(h, 2)
    if n % 2 == 0: return base-Fr(89, 36)+Fr(13, 6*n)-Fr(1, 18*(n-2))
    return base-Fr(73, 36)-Fr(10, 9*(n-1))+Fr(7, 6*(n-3))
def EU_Irho(n):
    if n < 4: return Fr(0)
    m = n//2
    w = b(m)*b(m-1) if n % 2 == 0 else b(m)**2
    return (m-Fr(w, 2*Cat(n-1)))/(m-1)
def EU_Irho2(n):
    if n < 4: return Fr(0)
    m = n//2
    if n % 2 == 0: return (n*n-Fr(4**(n-1), Cat(n-1)))/(4*(m-1)**2)
    return ((n-1)**2-Fr(4**(n-1)-2*b(m)**2, Cat(n-1)))/(4*(m-1)**2)
def VU_Irho(n): return EU_Irho2(n)-EU_Irho(n)**2
def EU_Irhop(n): return corr(n)*EU_Irho(n)
def VU_Irhop(n): return corr(n)**2*VU_Irho(n)
def EY_Irho(n): return Fr(0) if n < 4 else (Fr(1, 2) if n % 2 else Fr(n, 2*(n-1)))
def VY_Irho(n):   # corrected version of survey Prop. 58
    if n < 4: return Fr(0)
    return Fr(n+1, 12*(n-3)) if n % 2 else Fr(n*(n*n-2*n+4), 12*(n-2)*(n-1)**2)
def VY_Irhop(n):
    if n < 4: return Fr(0)
    return Fr(n+1, 12*(n-3)) if n % 2 else Fr(n*n-2*n+4, 12*n*(n-2))
def EU_sumI(n): return sum((NU(n, k)*EU_Irho(k) for k in range(4, n+1)), Fr(0))
def EU_sumIp(n): return sum((NU(n, k)*EU_Irhop(k) for k in range(4, n+1)), Fr(0))

def split_probs(model, k):
    if model == 'Y': return [Fr(1, k-1)]*(k-1)
    return [Fr(Cat(i-1)*Cat(k-i-1), Cat(k-1)) for i in range(1, k)]
def var_formula(model, t, a, n):
    """Theorem (law of total variance): Var X_n = sum_k N_n(k) w_k,
       w_k = E[(t(I,k-I)+a_I+a_{k-I})^2] - a_k^2, I = left root-subtree size."""
    N = NY if model == 'Y' else NU
    s = Fr(0)
    for k in range(2, n+1):
        p = split_probs(model, k)
        w = sum(p[i-1]*(t(i, k-i)+a[i]+a[k-i])**2 for i in range(1, k))-a[k]**2
        s += N(n, k)*w
    return s
MEAN = {('Y', 'J'): EY_J, ('U', 'J'): EU_J, ('Y', 'I'): EY_sumI, ('U', 'I'): EU_sumI,
        ('Y', 'Ip'): EY_sumIp, ('U', 'Ip'): EU_sumIp}
def means(model, name, n): return [Fr(0)]+[MEAN[model, name](i) for i in range(1, n+1)]

# ---------------- independent recursions ----------------
def unif_rec(t, N):
    T1 = [Fr(0)]*(N+1); T2 = [Fr(0)]*(N+1)
    for k in range(2, N+1):
        s1 = s2 = Fr(0)
        for i in range(1, k):
            j = k-i; ci, cj = Cat(i-1), Cat(j-1); x = t(i, j)
            s1 += x*ci*cj+T1[i]*cj+ci*T1[j]
            s2 += x*x*ci*cj+T2[i]*cj+ci*T2[j]+2*x*(T1[i]*cj+ci*T1[j])+2*T1[i]*T1[j]
        T1[k], T2[k] = s1, s2
    E = [Fr(0)]+[T1[k]/Cat(k-1) for k in range(1, N+1)]
    return E, [Fr(0)]+[T2[k]/Cat(k-1)-E[k]**2 for k in range(1, N+1)]
def yule_rec(t, N):
    E = [Fr(0)]*(N+1); M = [Fr(0)]*(N+1)
    for k in range(2, N+1):
        s1 = s2 = Fr(0)
        for i in range(1, k):
            j = k-i; x = t(i, j)
            s1 += x+E[i]+E[j]
            s2 += x*x+M[i]+M[j]+2*x*(E[i]+E[j])+2*E[i]*E[j]
        E[k], M[k] = s1/(k-1), s2/(k-1)
    return E, [M[k]-E[k]**2 for k in range(N+1)]

# ---------------- brute force ----------------
def trees(labels):
    if len(labels) == 1:
        yield 1; return
    f, rest = labels[0], labels[1:]
    for r in range(len(rest)):
        for S in combinations(rest, r):
            R = tuple(x for x in rest if x not in S)
            for tl in trees((f,)+S):
                for tr in trees(R):
                    yield (tl, tr)
def tree_stats(t):
    nodes = []
    def rec(t):
        if t == 1: return 1
        a = rec(t[0]); c = rec(t[1]); nodes.append((a+c, a, c)); return a+c
    rec(t)
    J = sum(1 for (nv, a, c) in nodes if a != c)               # Rogers J
    Iv, Ipv = [], []
    for (nv, a, c) in nodes:
        if nv >= 4:
            ce = -(-nv//2)
            I = Fr(max(a, c)-ce, (nv-1)-ce)                       # I_v
            Iv.append(I); Ipv.append(I if nv % 2 else Fr(nv-1, nv)*I)
    W = 1
    for (nv, a, c) in nodes: W *= nv-1                           # Yule weight denominator
    rootbig = bool(nodes) and nodes[-1][0] >= 4
    return (J, sum(Iv, Fr(0)), sum(Ipv, Fr(0)), Iv[-1] if rootbig else Fr(0),
            Ipv[-1] if rootbig else Fr(0)), W
NAMES = ['J', 'I', 'Ip', 'Irho', 'Irhop']
def brute(n):
    cnt = Counter(tree_stats(t) for t in trees(tuple(range(n))))
    tot = sum(cnt.values()); assert tot == (factorial(2*n-2)//(2**(n-1)*factorial(n-1)) if n > 1 else 1)
    py = lambda W: Fr(2**(n-1), factorial(n)*W)
    assert sum(c*py(W) for (s, W), c in cnt.items()) == 1
    res = {}
    for idx, nm in enumerate(NAMES):
        Eu = Fr(sum(c*s[idx] for (s, W), c in cnt.items()))/tot
        Vu = Fr(sum(c*s[idx]**2 for (s, W), c in cnt.items()))/tot-Eu**2
        Ey = sum(c*s[idx]*py(W) for (s, W), c in cnt.items())
        Vy = sum(c*s[idx]**2*py(W) for (s, W), c in cnt.items())-Ey**2
        res[nm] = (Eu, Vu, Ey, Vy)
    return res, tot

if __name__ == '__main__':
    T0 = time.time()
    # (1) brute force
    for n in range(1, NMAX_BRUTE+1):
        r, tot = brute(n)
        for nm in ['J', 'I', 'Ip']:
            aU, aY = means('U', nm, n), means('Y', nm, n)
            got = (aU[n], var_formula('U', TOLL[nm], aU, n), aY[n], var_formula('Y', TOLL[nm], aY, n))
            assert got == r[nm], (nm, n, got, r[nm])
        assert r['Ip'][3] == VY_sumIp(n)
        assert r['Irho'] == (EU_Irho(n), VU_Irho(n), EY_Irho(n), VY_Irho(n)), n
        assert r['Irhop'][:2] == (EU_Irhop(n), VU_Irhop(n)) and r['Irhop'][3] == VY_Irhop(n), n
        print(f'OK brute n={n} ({tot} labelled trees): all formulas match '
              f'[E_U J={r["J"][0]}, Var_U J={r["J"][1]}, E_Y J={r["J"][2]}, Var_Y J={r["J"][3]}]')
    # (2) recursions
    N = 60
    for nm in ['J', 'I', 'Ip']:
        EU, VU = unif_rec(TOLL[nm], N); EY, VY = yule_rec(TOLL[nm], N)
        aU, aY = means('U', nm, N), means('Y', nm, N)
        assert aU == EU and aY == EY, nm
        for n in range(1, 31):
            assert var_formula('U', TOLL[nm], aU, n) == VU[n] and var_formula('Y', TOLL[nm], aY, n) == VY[n], (nm, n)
        print(f'OK recursion {nm}: means n<=60, variance formulas n<=30')
    EY, VY = yule_rec(tIp, 120)
    assert all(VY_sumIp(n) == VY[n] for n in range(1, 121)); print('OK closed form Var_Y[Sigma I\'] n<=120')
    for n in range(4, 200):
        p = split_probs('U', n)
        for t, E1, V1 in [(tI, EU_Irho, VU_Irho), (tIp, EU_Irhop, VU_Irhop)]:
            e = sum(p[i-1]*t(i, n-i) for i in range(1, n)); e2 = sum(p[i-1]*t(i, n-i)**2 for i in range(1, n))
            assert e == E1(n) and e2-e*e == V1(n), n
        q = [Fr(1, n-1)]*(n-1)
        e = sum(q[i-1]*tI(i, n-i) for i in range(1, n)); e2 = sum(q[i-1]*tI(i, n-i)**2 for i in range(1, n))
        assert e == EY_Irho(n) and e2-e*e == VY_Irho(n)
    print('OK root I / I\' closed forms (uniform and Yule) n<=199')
    # (3) symbolic telescoping proof of the Var_Y[Sigma I'] closed form for n>=7
    import sympy as sp
    n_, Hn, Hh, H2 = sp.symbols('n H_n H_h H2_h'); R = sp.Rational
    def Vc(n, Hn, Hh, H2, par):
        base = R(33359, 15120)*n-R(4, 9)*n*(Hn-Hh)-R(13, 12)*n*H2
        return base-R(89, 36)+R(13, 6)/n-R(1, 18)/(n-2) if par == 0 else base-R(73, 36)-R(10, 9)/(n-1)+R(7, 6)/(n-3)
    def wc(k, par):
        return (k**2+3*k+38)/(12*(k-1)*(k-3)) if par else (k**3+48*k-52)/(12*k*(k-1)*(k-2))
    for par in (0, 1):
        m = (n_ if par == 0 else n_-1)/2
        V0 = Vc(n_, Hn, Hh, H2, par)
        V1 = Vc(n_+1, Hn+1/(n_+1), Hh, H2, 1) if par == 0 else Vc(n_+1, Hn+1/(n_+1), Hh+1/(m+1), H2+1/(m+1)**2, 0)
        expr = (V1-wc(n_+1, 1-par))/(n_+1)-(V0-wc(n_, par))/n_-2*wc(n_, par)/(n_*(n_+1))
        assert sp.simplify(expr) == 0
    aY = means('Y', 'Ip', 30)
    for k in range(7, 31):
        w = sum(Fr(1, k-1)*(tIp(i, k-i)+aY[i]+aY[k-i])**2 for i in range(1, k))-aY[k]**2
        assert w == Fr(k*k+3*k+38, 12*(k-1)*(k-3)) if k % 2 else w == Fr(k**3+48*k-52, 12*k*(k-1)*(k-2))
    print('OK symbolic telescoping identity + w_k closed form (k>=7) for Var_Y[Sigma I\']')
    # (4) asymptotics (numerical)
    n = 3000
    print('E_U[J]-(2-4/pi)n at n=3000:', float(EU_J(n))-(2-4/pi)*n, ' predicted -3/2+1/pi =', -1.5+1/pi)
    print('E_Y[J]-(2-2log2)n at n=3001:', float(EY_J(3001))-(2-2*log(2))*3001, ' predicted -1')
    print('E_Y[SumI]-(log2-5/12)n at n=3001:', float(EY_sumI(3001))-(log(2)-5/12)*3001, ' predicted -1/2')
    s = float(VY_sumIp(4001))/4001; c = 33359/15120-4/9*log(2)-13*pi**2/72
    print(f'Var_Y[SumI\']/n at n=4001: {s:.6f}, limit {c:.9f}')
    for n in (400, 1600):
        print(f'n={n}: E_U[Irho]={float(EU_Irho(n)):.6f} vs 1-2/sqrt(pi n)={1-2/sqrt(pi*n):.6f};'
              f' sqrt(n)Var_U[Irho]={sqrt(n)*float(VU_Irho(n)):.5f} vs (4-pi)/sqrt(pi)={(4-pi)/sqrt(pi):.5f}')
    print(f'done in {time.time()-T0:.0f}s')
