from fractions import Fraction as Fr
from math import comb
Cat=lambda m: comb(2*m,m)//(m+1)
def a(n): return 1 if n==0 else sum(Fr(k,n)*comb(2*n-k-1,n-1)*2**(n-k) for k in range(1,n+1))
print([a(n) for n in range(10)])
def E(n): return Fr(3*(n-1),n+1)
def V(n): return Fr(Cat(n-1)-Cat(n)+Cat(n+1),Cat(n-1)) - Fr(4*a(n),2**n*Cat(n-1)) - E(n)**2
print([str(V(n)) for n in range(1,14)])
print(float(V(2000)), 4/9)
