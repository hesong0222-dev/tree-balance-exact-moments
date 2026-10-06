"""Numerical asymptotics (natural log) of the sb-shape moments, evaluated from the exact
finite-sum formulas (Theorems in section.tex) with FFT convolutions. Floats only."""
import numpy as np
from math import log, pi, sqrt
from scipy.signal import fftconvolve

def scaled(N):
    """ct[m]=C_m/4^m, bt[m]=b_m/4^m for m=0..N."""
    m = np.arange(1, N+1)
    bt = np.concatenate([[1.0], np.cumprod((2*m-1)/(2*m))])
    ct = bt/np.arange(1, N+2)
    return ct, bt

def uniform(n):
    ct, bt = scaled(n+1)
    k = np.arange(n+1); l = np.zeros(n+1); l[2:] = np.log(k[2:]-1)
    a = np.zeros(n+1); a[2:] = ct[k[2:]-1]*l[2:]               # a_j = c~_{j-1} l_j
    conv_ab = fftconvolve(a, bt[:n+1])[:n+1]                    # sum_j a_j b~_{k-j}
    mu = np.zeros(n+1); mu[2:] = conv_ab[2:]/ct[k[2:]-1]          # mu_k = E_U[s_k]
    en = np.zeros(n+1); en[2:] = ct[k[2:]-1]*bt[n-k[2:]]/ct[n-1]  # e_n(k)
    S2 = (en*l*(2*mu-l)).sum()
    aa = fftconvolve(a, a)[:n+1]                                # sum_{j+k=s} a_j a_k
    s = np.arange(n+1); mm = n-s+2
    w = np.where(s >= 4, (mm-1)*bt[np.clip(mm-1, 0, n)]/ct[n-1], 0.0)
    S2 += (w*aa).sum()
    return mu[n], S2-mu[n]**2

def yule(n):
    """Var via the split recursion Var_n = s_n + 2n sum_{k<n} s_k/(k(k+1)) (uniform root split)."""
    k = np.arange(n+1, dtype=float); l = np.zeros(n+1); l[2:] = np.log(k[2:]-1)
    # mu_k = l_k + 2k sum_{j<k} l_j/(j(j+1))
    t = np.zeros(n+1); t[2:] = 2*l[2:]/(k[2:]*(k[2:]+1))
    cs = np.concatenate([[0.0], np.cumsum(t)[:-1]])            # sum_{j<k}
    mu = l + k*cs; mu[1] = 0.0; mu[0] = 0.0
    sq = np.zeros(n+1); sq[1:] = np.cumsum(mu[1:]**2)            # for sum_i mu_i^2
    mm = fftconvolve(mu, mu)[:n+1]                               # sum_{i} mu_i mu_{k-i} (i=0..k, mu_0=0)
    sk = np.zeros(n+1)
    kk = np.arange(2, n+1)
    sk[2:] = (2*sq[kk-1] + 2*mm[kk])/(kk-1) - (mu[kk]-l[kk])**2
    acc = np.concatenate([[0.0], np.cumsum(np.where(k >= 2, sk/(k*(k+1)+1e-300), 0.0))])
    var = sk[n] + 2*n*acc[n]                                     # acc[n] = sum_{k<n}
    return mu[n], var

M = 4_000_000
m = np.arange(1, M, dtype=float)
KY = (2*np.log(m)/((m+1)*(m+2))).sum() + 2*(log(M)+1)/M
ct, _ = scaled(M)
C0 = (np.log(m)*ct[1:M]).sum() + 2*(log(M)+2)/sqrt(pi*M)
print(f'K_Y = 2 sum ln m/((m+1)(m+2)) = {KY:.9f};  C_0 = sum ln m C_m 4^-m = {C0:.7f}')
for n in (1000, 10000, 100000, 1000000):
    mY, vY = yule(n); mU, vU = uniform(n)
    print(f'n={n:>7}: E_Y-(K_Y n-ln n-2)={mY-KY*n+log(n)+2: .2e}  Var_Y/n={vY/n:.6f}  '
          f'(E_U-C_0 n)/sqrt(n)={(mU-C0*n)/sqrt(n):.4f} [-2sqrt(pi)={-2*sqrt(pi):.4f}]  Var_U/(n ln n)={vU/(n*log(n)):.4f}')
# local slope of Var_U/n against ln n
prev = None
for n in (2**14, 2**16, 2**18, 2**20, 2**22):
    _, vU = uniform(n)
    if prev: print(f'slope d(Var_U/n)/d(ln n) between n/4 and n={n}: {(vU/n-prev)/log(4):.4f}  [8(1-ln 2)={8*(1-log(2)):.4f}]')
    prev = vU/n
