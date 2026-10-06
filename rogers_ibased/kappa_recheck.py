import sys
from mpmath import mp, mpf, sqrt, log, catalan, pi, nstr
from fractions import Fraction as Fr
from math import comb
mp.dps=90
def Cat(m): return comb(2*m,m)//(m+1)
def b(m): return comb(2*m,m)
# exact reference T(k) = 4*C_{k-1}*iota_k/4^k from the definition (independent of any closed form)
def T_exact(k,prime=False):
    m=-(-k//2); s=Fr(0)
    for i in range(1,k):
        I=Fr(max(i,k-i)-m,(k-1)-m)
        if prime and k%2==0: I*=Fr(k-1,k)
        s+=Cat(i-1)*Cat(k-i-1)*I
    return 4*s/Fr(4**k)
Mmax=int(sys.argv[1]) if len(sys.argv)>1 else 1800000
# term recurrences, normalized: c(j)=C_j/4^j, be(m)=b_m/4^m
c=[mpf(1)]  # c(0)
def cget(j):
    while len(c)<=j:
        jj=len(c); c.append(c[-1]*(2*jj-1)/(2*(jj+1)))
    return c[j]
checkpoints=sorted(set(int(2000*(j+1)**2) for j in range(30) if 2000*(j+1)**2<=Mmax))
S=mpf(0); Sp=mpf(0); be_prev=mpf(1) # be(m-1) with m starting at 2 -> be(1)=1/2
be_m1=mpf(1)/2
part={}
cj2=cget(2)  # c(2m-2) for m=2
cj3=None
for m in range(2,Mmax+1):
    be_m=be_m1*(2*m-1)/(2*m)
    if m==2: c2m2=cget(2); c2m1=cget(3); c2m=cget(4)
    else:
        c2m2=c2m; c2m1=c2m2*(2*(2*m-1)-1)/(2*(2*m-1+1)); c2m=c2m1*(2*(2*m)-1)/(2*(2*m+1))
    ve=(2*(4*m-3)*c2m2/16 - be_m*be_m1/4)*4/(2*(m-1))
    vo=(2*(4*m-1)*c2m1/16 - be_m*be_m/4 - c2m/4)*4/(2*(m-1))
    if m<=8:
        assert abs(ve-mpf(T_exact(2*m).numerator)/T_exact(2*m).denominator)<mpf(10)**-80
        assert abs(vo-mpf(T_exact(2*m+1).numerator)/T_exact(2*m+1).denominator)<mpf(10)**-80
        tp=T_exact(2*m,True); assert abs(ve*(2*m-1)/(2*m)-mpf(tp.numerator)/tp.denominator)<mpf(10)**-80
    S+=ve+vo; Sp+=ve*(2*m-1)/(2*m)+vo
    be_m1=be_m
    if m in checkpoints: part[m]=(S,Sp)
print('term recurrences match definition for k<=17', flush=True)
# Neville extrapolation in h=M^{-1/2} -> 0
def extrap(pts):
    xs=[mpf(M)**-0.5 for M,_ in pts]; ys=[v for _,v in pts]; n=len(xs); P=ys[:]
    for k in range(1,n):
        for i in range(n-k):
            P[i]=((0-xs[i+k])*P[i]+(xs[i]-0)*P[i+1])/(xs[i]-xs[i+k])
    return P[0]
Ms=sorted(part)
s2=sqrt(2); L=log(1+s2); G=catalan
kcf=(11*s2-10-9*L)/12+(6*G-1)/(2*pi); kpcf=(4*s2-5-12*L)/24+(2*G+1)/pi
for use in [len(Ms)-4,len(Ms)-2,len(Ms)]:
    sel=Ms[:use]
    k=extrap([(M,part[M][0]) for M in sel]); kp=extrap([(M,part[M][1]) for M in sel])
    print(f'{use} pts (Mmax={sel[-1]}): kappa={nstr(k,36)}  diff={nstr(k-kcf,3)} | kappa\'={nstr(kp,36)} diff={nstr(kp-kpcf,3)}',flush=True)
print('closed kappa =',nstr(kcf,36)); print("closed kappa'=",nstr(kpcf,36))
