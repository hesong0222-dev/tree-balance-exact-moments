# Colless index under uniform/PDA model: exact checks
from fractions import Fraction as Fr
from math import comb
from itertools import combinations
def Cat(m): return comb(2*m,m)//(m+1) if m>=0 else 0
N=60
# a_n = sum_{i+j=n} |i-j| C_{i-1} C_{j-1}
a=[0]*(N+1)
for n in range(2,N+1): a[n]=sum(abs(2*i-n)*Cat(i-1)*Cat(n-i-1) for i in range(1,n))
def a_closed(n):
    h=(n+1)//2
    return 2*(2*n-3)*Cat(n-2)-2*h*(2*n-2*h-1)*Cat(h-1)*Cat(n-h-1)
assert all(a[n]==a_closed(n) for n in range(2,N+1)), 'a closed form fails'
# tot_n = sum over plane binary trees with n leaves of Colless, via recursion
tot=[0]*(N+1)
for n in range(2,N+1):
    tot[n]=sum(abs(2*i-n)*Cat(i-1)*Cat(n-i-1)+tot[i]*Cat(n-i-1)+tot[n-i]*Cat(i-1) for i in range(1,n))
tot_conv=[sum(a[k]*comb(2*(n-k),n-k) for k in range(2,n+1)) for n in range(N+1)]
assert tot==tot_conv
# brute force: all labeled rooted binary trees (PDA uniform) for n<=8
def trees(labels):
    labels=tuple(labels)
    if len(labels)==1: yield 1; return
    first,rest=labels[0],labels[1:]
    for r in range(0,len(rest)):
        for S in combinations(rest,r):
            L=(first,)+S; R=tuple(x for x in rest if x not in S)
            for tl in trees(L):
                for tr in trees(R):
                    yield (tl,tr)
def col(t):
    if t==1: return 1,0
    (nl,cl),(nr,cr)=col(t[0]),col(t[1]); return nl+nr, cl+cr+abs(nl-nr)
for n in range(2,9):
    vals=[col(t)[1] for t in trees(range(n))]
    E=Fr(sum(vals),len(vals)); V=Fr(sum(v*v for v in vals),len(vals))-E*E
    assert E==Fr(tot[n],Cat(n-1)), n
    print(n,len(vals),E,V)
print('all checks pass; E_U for n=2..12:',[str(Fr(tot[n],Cat(n-1))) for n in range(2,13)])
