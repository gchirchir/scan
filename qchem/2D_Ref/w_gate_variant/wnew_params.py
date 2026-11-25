"""Proposed new weight (2026-11-24): W_new = W_orig * p^k / (p^k + p0^k), W_orig = jana_ref Eq. (4) unchanged
(constant 1e-10, i.e. t = 1e-5 in g2(alpha p / t)).  Tuple (t, p0, k) as used by verify_variant.weight."""
NEW = (1e-5, 6.0, 12)
ORIG = (1e-5, 0, 0)
ALT = (1e-3, 5.0, 12)      # earlier variant: constant changed to t^2 = 1e-6, p0 = 5 (kept as alternative)
ATOMIC_KEY = "Jana t + p0=6 k=12"   # column prefix in w_gate_variant/atomic_variant_v2.csv
