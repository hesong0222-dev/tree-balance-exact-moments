import numpy as np
from math import pi, log, sqrt, lgamma, exp
from scipy.signal import fftconvolve
M=200001
j=np.arange(M)
# beta_j = b_j/4^j via recurrence
beta=np.empty(M); beta[0]=1.0
for k in range(1,M): beta[k]=beta[k-1]*(2*k-1)/(2*k)
# e'_j/4^j = b_{floor(j/2)} b_{ceil(j/2)} / 4^j = beta[fl]*beta[ce]
ep=beta[j//2]*beta[(j+1)//2]
S=fftconvolve(ep,beta)[:M]   # S_N/4^N
g=0.5772156649015329
c2=(2/pi)*(g+5*log(2)-1)
for n in [100,1000,10000,100000,200000]:
    N=n-1
    CN_over=beta[N]/(N+1)  # C_N/4^N
    E=(1-S[N])/CN_over/4  # tot_n/C_{n-1} = (4^{N}-S_N)/C_N ; careful: tot_n=4^{n-1}-S_{n-1}=4^N(1-S[N])
    E=(1-S[N])/CN_over
    r=E-(sqrt(pi)*n**1.5-(2/pi)*n*log(n)-c2*n)
    print(n, E, r/sqrt(n))
