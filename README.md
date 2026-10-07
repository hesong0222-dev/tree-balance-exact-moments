# Supplementary code: Exact moments of tree balance indices under the Yule and uniform models

Python 3 with sympy, mpmath, numpy, scipy (`pip install sympy mpmath numpy scipy`).
Run any script with `python3 <script>` from its folder. Each script prints OK lines or raises an assertion.

- verify_all.py   Colless (uniform): Gosper lemma (symbolic), Theorems on mean and variance vs brute force over all labelled trees, n<=9
- check.py, asym.py   Colless mean recursion n<=60; asymptotic expansion (FFT, n<=2e5)
- colless_large.py   Colless (uniform): mean and second-moment formulas vs root-split recursion in exact rationals, n<=60; six-term expansion at n=10^6 and 2*10^6 in 40-digit arithmetic (about 30 s)
- halpha.py, b2final.py   H_alpha / B2 moments under the uniform model vs plane-tree enumeration
- i2/verify.py   equal-weights Colless I2 (Yule, uniform)
- vld/verify.py   variance of leaf depths (Yule, uniform)
- rogers_ibased/verify.py   Rogers J, I-based indices; kappa_recheck.py: 70-digit check of the kappa conjecture
- sshape_b2y/verify.py   sb-shape statistic; B2 variance under the Yule model

Paper: H. Song, *Exact moments of tree balance indices under the Yule and uniform models* (2026), arXiv link to be added. Archived code: doi:10.5281/zenodo.23181608.

License: MIT.
