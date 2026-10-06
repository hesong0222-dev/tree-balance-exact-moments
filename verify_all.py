"""Colless index, uniform/PDA model: verify Theorem 1 (E), Lemma (Gosper), Theorem 2 (Var)."""
from fractions import Fraction as Fr
from math import comb
from itertools import combinations
from sympy import symbols, binomial, combsimp, simplify
# --- Lemma: T_{i+1}-T_i = (2i-n) C_{i-1} C_{n-i-1}, T_i = i(2n-2i-1) C_{i-1} C_{n-i-1}
i,n=symbols('i n',integer=True,positive=True)
Ct=lambda x: binomial(2*x,x)/(x+1)
T=lambda x: x*(2*n-2*x-1)*Ct(x-1)*Ct(n-x-1)
d=combsimp((T(i+1)-T(i))/((2*i-n)*Ct(i-1)*Ct(n-i-1)))
print('Lemma ratio (must be 1):',simplify(d))
def Cat(m): return comb(2*m,m)//(m+1) if m>=0 else 0
def b(m): return comb(2*m,m) if m>=0 else 0
def S(N): return sum(b(j//2)*b((j+1)//2)*b(N-j) for j in range(N+1))
def tot(n): return 4**(n-1)-S(n-1)               # Theorem 1 numerator
def E(n): return Fr(tot(n),Cat(n-1))
def M2(n):                                        # Theorem 2
    s=0
    for k in range(2,n+1):
        Q=k*k*Cat(k-1)-4**(k-1)
        P=sum(tot(a)*tot(k-a) for a in range(1,k))
        H=sum(abs(2*a-k)*tot(a)*Cat(k-a-1) for a in range(1,k))
        s+=b(n-k)*(Q+2*P+4*H)
    return Fr(s,Cat(n-1))
def V(n): return M2(n)-E(n)**2
# --- brute force over all labeled rooted binary trees (PDA = uniform on these)
def trees(labels):
    if len(labels)==1: yield 1; return
    f,rest=labels[0],labels[1:]
    for r in range(len(rest)):
        for Sx in combinations(rest,r):
            Lt=(f,)+Sx; Rt=tuple(x for x in rest if x not in Sx)
            for tl in trees(Lt):
                for tr in trees(Rt): yield (tl,tr)
def col(t):
    if t==1: return 1,0
    (a,c1),(bb,c2)=col(t[0]),col(t[1]); return a+bb,c1+c2+abs(a-bb)
for m in range(2,10):
    vals=[col(t)[1] for t in trees(tuple(range(m)))]
    Eb=Fr(sum(vals),len(vals)); Vb=Fr(sum(v*v for v in vals),len(vals))-Eb**2
    assert (Eb,Vb)==(E(m),V(m)), m
    print(f'n={m}: trees={len(vals)} E={Eb} Var={Vb}  OK')
