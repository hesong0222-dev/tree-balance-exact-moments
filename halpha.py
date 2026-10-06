# Exact E, Var of W=sum_leaves q^depth (q=2^-alpha) under PDA; H_alpha=(1-W)/(1-2^{1-alpha})
from fractions import Fraction as Fr
from math import comb
Cat=lambda m: comb(2*m,m)//(m+1)
def beta(m,s):
    if m==0: return 1 if s==0 else 0
    if s==0: return 0
    return Fr(s,m)*comb(2*m-s-1,m-1) if s<=m else 0
def EW(n,q): return sum((2*q)**s*beta(n-1,s) for s in range(n))/Cat(n-1)
def EW2(n,q):
    g=lambda s: sum((j+1)*(2*q)**j*(2*q*q)**(s-j) for j in range(s+1))
    t=sum((2*q*q)**s*beta(n-1,s) for s in range(n))
    t+=2*q*q*sum(beta(n-2,s)*g(s) for s in range(n-1)) if n>=2 else 0
    return t/Cat(n-1)
# brute force over plane binary trees (uniform plane = PDA for shape functionals)
from functools import lru_cache
@lru_cache(None)
def plane(n):
    if n==1: return [ (0,) ]  # list of leaf-depth tuples
    out=[]
    for i in range(1,n):
        for L in plane(i):
            for R in plane(n-i):
                out.append(tuple(d+1 for d in L+R))
    return out
for q in [Fr(1,4),Fr(1,8),Fr(1,3),Fr(1,2)]:
    for n in range(1,11):
        ws=[sum(q**d for d in t) for t in plane(n)]
        m1=sum(ws,Fr(0))/len(ws); m2=sum((w*w for w in ws),Fr(0))/len(ws)
        assert m1==EW(n,q) and m2==EW2(n,q),(q,n)
print('H_alpha moments: plane-tree brute force n<=10, q in {1/4,1/8,1/3,1/2} OK')
# B2 direct formula vs plane brute force n<=12
def a(n): return 1 if n==0 else sum(Fr(k,n)*comb(2*n-k-1,n-1)*2**(n-k) for k in range(1,n+1))
def VB2(n): return Fr(Cat(n-1)-Cat(n)+Cat(n+1),Cat(n-1)) - Fr(4*a(n),2**n*Cat(n-1)) - Fr(3*(n-1),n+1)**2
for n in range(1,13):
    xs=[sum(Fr(d,2**d) for d in t) for t in plane(n)]
    m1=sum(xs,Fr(0))/len(xs); v=sum((x*x for x in xs),Fr(0))/len(xs)-m1*m1
    assert m1==Fr(3*(n-1),n+1) and v==VB2(n), n
print('B2 E,Var: plane brute force n<=12 OK')
# limit check vs Corollary 2.24 (alpha=2)
import math
al=2; q=Fr(1,4); n=300
V=(EW2(n,q)-EW(n,q)**2)/(1-Fr(2)**(1-al))**2
lim=2**-(2*al+1)/((1-2**(-2*al))**2*(1-2**(1-al))**2)*(1+(2**(1-2*al)-1)/(2*(1-2**-al)**2))
print('alpha=2 Var n=300:',float(V),' BDM limit:',lim)
