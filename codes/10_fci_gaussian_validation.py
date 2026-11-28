#!/usr/bin/env python3
"""FCI validation of the fixed two-electron Gaussian exchange reference.

Run from the project root with the PySCF environment used by codes 03 and 08::

    $HOME/miniconda3/envs/anigto/bin/python codes/10_fci_gaussian_validation.py

This is an independent validation script; it does not import codes 01--09.
For the anisotropic harmonic trap

    h = -1/2 nabla^2 + 1/2 sum_k omega_k^2 r_k^2,

the normalized ground orbital is the Gaussian used in the manuscript.  With
omega=(1,1,lambda^2) (sheet) or (lambda^2,lambda^2,1) (wire), the one-particle
ground-state matrix element is h00 = (omega_x+omega_y+omega_z)/2.  The
one-spatial-orbital singlet FCI problem has

    E_FCI = 2 h00 + J,       E_x^HF = -J,

where J=(phi phi|phi phi).  The exact exchange is therefore recovered from
the FCI repulsion contribution, not from the correlated total energy itself.

For n_A = 2 sqrt(det A) pi^(-3/2) exp(-r^T A r), Eq. (7) is evaluated as

    E_x = -(2/sqrt(pi)) int_0^inf
           sqrt(det(A/2) / det(A/2 + t^2 I)) dt.

The logarithmic t=exp(s) transformation keeps the integral stable for the
lambda=10^4 wire. Outputs are written to codes/results/ and reproduce the
sheet and wire exact columns in the existing benchmark tables.

SCAN and r2SCAN exchange (added 2026-11-17, coauthor request) are evaluated
on the same fixed densities with Libxc (MGGA_X_SCAN, MGGA_X_R2SCAN through
PySCF) on the logarithmic cylindrical Gauss-Legendre grid of codes 01/02
(functions copied verbatim).  For one doubly occupied orbital tau = tau_W, so
alpha = alpha_bar = 0 and both functionals reduce to F_x = 1.174 g_x(s) with
the same g_x; r2SCAN exchange therefore equals SCAN exchange on these
densities, and DC-r2SCAN equals DC-SCAN.  The script checks this numerically
and against the closed-form alpha=0 SCAN enhancement.  Exchange-only table
bodies: tab_fci_{sheet,wire}_exchange_rows.tex; DFA comparison:
tab_r2scan_{sheet,wire}_rows.tex.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from pyscf import fci
from pyscf.dft import libxc
from scipy.integrate import quad
from scipy.special import gamma


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LAMBDAS = (1.0, 2.0, 10.0, 100.0, 1000.0, 10000.0)


def trap_omega(family: str, lam: float) -> np.ndarray:
    """Return omega for the sheet or wire harmonic trap."""
    if family == "sheet":
        return np.array((1.0, 1.0, lam * lam), dtype=float)
    if family == "wire":
        return np.array((lam * lam, lam * lam, 1.0), dtype=float)
    raise ValueError(f"unknown family: {family}")


def exact_closed_form(family: str, lam: float) -> float:
    """Closed-form exact exchange used by the existing sheet/wire tables."""
    q = math.sqrt(max(lam * lam - 1.0, 0.0))
    if q < 1.0e-10:
        return -math.sqrt(2.0 / math.pi)
    if family == "sheet":
        return -math.sqrt(2.0 / math.pi) * lam * math.atan(q) / q
    if family == "wire":
        return -math.sqrt(2.0 / math.pi) * lam * math.acosh(lam) / q
    raise ValueError(f"unknown family: {family}")


def exact_eq7(omega: np.ndarray) -> tuple[float, float]:
    """Evaluate the negative exchange in Eq. (7)."""
    b = 0.5 * np.asarray(omega, dtype=float)
    log_b = np.log(b)
    breakpoints = np.log(np.sqrt(b))
    lo = float(np.min(breakpoints) - 40.0)
    hi = float(np.max(breakpoints) + 40.0)
    points = sorted({float(p) for p in breakpoints if lo < p < hi})

    def log_integrand(s: float) -> float:
        # t = exp(s), with the determinant ratio evaluated in log space.
        return s + 0.5 * (float(np.sum(log_b)) - float(np.sum(np.logaddexp(log_b, 2.0 * s))))

    def integrand(s: float) -> float:
        return math.exp(log_integrand(s))

    integral, error = quad(
        integrand,
        lo,
        hi,
        points=points,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=500,
    )
    exchange = -2.0 * integral / math.sqrt(math.pi)
    return exchange, 2.0 * error / math.sqrt(math.pi)


def fci_one_orbital(omega: np.ndarray, j: float) -> tuple[float, float, float, float, float]:
    """Run singlet FCI in the one-orbital space and extract J and exchange."""
    h00 = 0.5 * float(np.sum(omega))
    h1 = np.array([[h00]], dtype=float)
    eri = np.array([[[[j]]]], dtype=float)
    e_fci, _ = fci.direct_spin0.kernel(h1, eri, 1, 2, verbose=0)
    j_fci = float(e_fci - 2.0 * h00)
    ex_fci = -j_fci
    t00 = 0.25 * float(np.sum(omega))
    v_harm = 0.25 * float(np.sum(omega))
    return h00, t00, v_harm, float(e_fci), ex_fci


# ==============================================================================================================
# SCAN / r2SCAN / DC exchange on the fixed Gaussian densities (copied verbatim from codes/01_wire_benchmark.py)
# ==============================================================================================================
CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)        # LDA exchange constant, E_x^LDA = -CX int n^(4/3)
KF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)            # k_F = KF n^(1/3)
C_NORM = 2.0 / np.pi ** 1.5                       # n = lam^k C_NORM exp(-(u + t^2)) in reduced coordinates

# SCAN exchange (Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)); alpha = 0 branch only is needed here.
SCAN_H0X = 1.174
SCAN_A1 = 4.9479
A1D = 1.0 / 3.0


def m2d():
    """M2D of Eq. (28): the sheet integral of n^(4/3) s^(-1/2) in the lam -> infinity limit."""
    # E_x -> -A2D M2D with  M2D = 2 pi CX c^(4/3) int du dt e^{-4/3(u+t^2)} (K n^{1/3})^{1/2} / |t|^{1/2}
    # (s -> lam^{2/3} |t| / (K n^{1/3}) with lam^{1/3} prefactor cancelling).  Closed form:
    # u and t integrals in closed form:
    a = 4.0 / 3.0 + 1.0 / 6.0                    # s^{-1/2} carries n^{1/6}: exp(-(u+t^2)/6)
    iu = 1.0 / a
    it = 0.5 * gamma(0.25) * a ** (-0.25)        # int_0^inf t^{-1/2} e^{-a t^2} dt
    return 2.0 * np.pi * CX * C_NORM ** (4.0 / 3.0) * np.sqrt(KF) * C_NORM ** (1.0 / 6.0) * iu * it


A2D = np.sqrt(np.pi / 2.0) / m2d()


def fx_scan_alpha0(s):
    """SCAN F_x(s, alpha = 0) = h0x g_x(s) = 1.174 (1 - exp(-a1/sqrt(s))), g_x(0) = 1 (Eq. 6)."""
    s = np.asarray(s, dtype=float)
    with np.errstate(divide='ignore'):
        return SCAN_H0X * -np.expm1(-SCAN_A1 / np.sqrt(np.where(s > 0, s, 1e-300)))


def f2d(s):
    return A2D / (1.0 + s * s) ** 0.25


def f1d(s):
    return A1D / (1.0 + s * s)


def geometry_weights(D):
    """Eqs. (23)-(24): Lagrange-type selectors of the 2D (D=1), 1D (D=2) and SCAN (D=3) branches."""
    l2 = 0.5 * (D - 2.0) * (D - 3.0)
    l1 = (D - 1.0) * (3.0 - D)
    l3 = 0.5 * (D - 1.0) * (D - 2.0)
    den = l2 * l2 + l1 * l1 + l3 * l3
    return l2 * l2 / den, l1 * l1 / den, l3 * l3 / den


def omega_switch(s, Z, variant='full'):
    """Eq. (25) ('full') or the minimal-information Eq. (37) ('min')."""
    s8 = s ** 8
    g = s8 / (1.0 + s8)
    g = np.where(np.isfinite(s8), g, 1.0)
    if variant == 'min':
        return g
    z3 = Z ** 3
    return z3 / (1.0 + z3) * g


def fx_dcscan(s, D, Z, fx_base, variant='full'):
    w2, w1, _ = geometry_weights(D)
    om = omega_switch(s, Z, variant)
    return (1.0 - om * (w2 + w1)) * fx_base + om * (w2 * f2d(s) + w1 * f1d(s))


def _log_gl(lo, hi, n):
    """Gauss-Legendre nodes in log coordinates: x in [lo, hi], returns x and weights for dx."""
    p, w = np.polynomial.legendre.leggauss(n)
    a, b = np.log(lo), np.log(hi)
    q = 0.5 * (b - a) * p + 0.5 * (b + a)
    x = np.exp(q)
    return x, 0.5 * (b - a) * w * x


def family_fields(family, lam, u, t):
    """s, D, Z on the reduced (u, t) grid of Eq. (33) for family 'sheet' or 'wire'.

    wire : u = lam^2 (x^2+y^2), t = z      -> r^T A^2 r = lam^2 u + t^2,  r^T A^4 r = lam^6 u + t^2
    sheet: u = x^2 + y^2,       t = lam z  -> r^T A^2 r = u + lam^2 t^2,  r^T A^4 r = u + lam^6 t^2
    """
    k = 2 if family == 'wire' else 1
    U, T = np.meshgrid(u, t, indexing='ij')
    T2 = T * T
    n = lam ** k * C_NORM * np.exp(-(U + T2))
    kf = KF * np.cbrt(n)
    if family == 'wire':
        a2 = lam ** 2 * U + T2
        a4 = lam ** 6 * U + T2
        trc2 = 2.0 * lam ** 4 + 1.0
    else:
        a2 = U + lam ** 2 * T2
        a4 = U + lam ** 6 * T2
        trc2 = 2.0 + lam ** 4
    s = np.sqrt(a2) / kf                       # |grad n| / (2 k_F n), |grad n|/n = 2 sqrt(r^T A^2 r)
    with np.errstate(invalid='ignore', divide='ignore'):
        D = np.where(a4 > 0, a2 * trc2 / a4, 3.0)
    D = np.clip(D, 1.0, 3.0)
    Z = np.sqrt(trc2) / kf ** 2
    return U, T2, s, D, Z


# --------------------------------------------------------------------------- Libxc SCAN / r2SCAN (this script)
LIBXC_X = {'LDA': 'LDA_X', 'SCAN': 'MGGA_X_SCAN', 'r2SCAN': 'MGGA_X_R2SCAN'}
SHEET_LIMIT_EXACT = -math.sqrt(math.pi / 2.0)
SHEET_LIMIT_SCAN = -SCAN_H0X * SCAN_A1 * m2d()      # lam -> inf; identical for r2SCAN (same h0x, a1, g_x)


def dfa_exchange(family, lam, n=400, u_range=(np.exp(-32.0), 26.0), t_range=(np.exp(-32.0), 8.0)):
    """Exchange-only energies of the fixed Gaussian density: Libxc LDA, SCAN, r2SCAN; closed-form alpha=0
    SCAN; DC-SCAN (closed-form base, = codes 01/02) and DC-r2SCAN (Libxc r2SCAN base).

    Reduced coordinates as in family_fields; d^3r = 2 pi du dt / lam^k (t > 0 half counted twice).
    Libxc input per point: n, grad n (|grad n| along x), laplacian, tau = tau_W = |grad n|^2 / (8 n).
    The energy uses the analytic LDA weight times F_x = e_x / e_x^LDA, as in code 03; points where Libxc
    returns zero (below its density threshold) keep the alpha=0 closed form (contribution < 1e-12 Ha).
    """
    k = 2 if family == 'wire' else 1
    u, wu = _log_gl(*u_range, n)
    t, wt = _log_gl(*t_range, n)
    U, T2, s, D, Z = family_fields(family, lam, u, t)
    rho = (lam ** k * C_NORM * np.exp(-(U + T2))).ravel()
    if family == 'wire':
        a2, tra = lam ** 2 * U + T2, 2.0 * lam ** 2 + 1.0
    else:
        a2, tra = U + lam ** 2 * T2, 2.0 + lam ** 2
    a2 = a2.ravel()
    gnorm = 2.0 * rho * np.sqrt(a2)                      # grad n = -2 n A r
    lap = rho * (4.0 * a2 - 2.0 * tra)
    tau = gnorm * gnorm / (8.0 * rho)
    zero = np.zeros_like(rho)
    inp = np.vstack((rho, gnorm, zero, zero, lap, tau))
    eps = {key: np.asarray(libxc.eval_xc(code + ',', inp if key != 'LDA' else rho, spin=0, deriv=0)[0])
           for key, code in LIBXC_X.items()}
    ok = eps['LDA'] != 0.0
    fs = fx_scan_alpha0(s).ravel()
    fx = {key: np.where(ok, eps[key] / np.where(ok, eps['LDA'], 1.0), fs) for key in ('SCAN', 'r2SCAN')}
    W = (np.outer(wu, wt) * np.exp(-4.0 / 3.0 * (U + T2))).ravel()
    pref = -2.0 * np.pi * CX * C_NORM ** (4.0 / 3.0) * lam ** (k / 3.0)
    s, D, Z = s.ravel(), D.ravel(), Z.ravel()
    # direct Libxc energy as an independent check of the F_x route
    vol = (2.0 * np.pi * np.outer(wu, wt) / lam ** k).ravel()
    return {
        'LDA': pref * W.sum(),
        'SCAN': pref * np.sum(W * fx['SCAN']),
        'r2SCAN': pref * np.sum(W * fx['r2SCAN']),
        'SCAN_closed_form': pref * np.sum(W * fs),
        'DC-SCAN': pref * np.sum(W * fx_dcscan(s, D, Z, fs, 'full')),
        'DC-r2SCAN': pref * np.sum(W * fx_dcscan(s, D, Z, fx['r2SCAN'], 'full')),
        'SCAN_direct': float(np.sum(vol * rho * eps['SCAN'])),
        'r2SCAN_direct': float(np.sum(vol * rho * eps['r2SCAN'])),
        'libxc_zero_points': int(np.sum(~ok)),
    }


def latex_exchange_rows(rows: list[dict[str, float | str]]) -> str:
    """Exchange-only FCI table body: FCI and SM Eq. (7) exchange, their difference, SCAN, r2SCAN, DC-SCAN, DC-r2SCAN.

    Sheet lam -> inf: DC-SCAN = DC-r2SCAN = -sqrt(pi/2) by construction (A2D fixed on that limit, Eq. 28-29).
    """
    lines = [
        "% generated by codes/10_fci_gaussian_validation.py (exchange only)",
        r"\begin{tabular}{llrrrrrrr}",
        r"family & $\lambda$ & $E_x^{\rm FCI}$ & SM Eq.~(7) & $\Delta E_x$ & $E_x^{\rm SCAN}$ & $E_x^{\mathrm{r}^2\mathrm{SCAN}}$ & $E_x^{\rm DC\text{-}SCAN}$ & $E_x^{\mathrm{DC\text{-}r}^2\mathrm{SCAN}}$ "
        + "\\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['family']} & {row['lambda']:.0f} & {row['ex_fci']:.9f} & {row['ex_eq7']:.9f} & "
            f"{row['ex_fci_minus_closed']:+.2e} & {row['ex_scan']:.6f} & {row['ex_r2scan']:.6f} & "
            f"{row['ex_dcscan']:.6f} & {row['ex_dcr2scan']:.6f}" + " \\\\",
        )
    if rows and rows[0]['family'] == 'sheet':
        lines.append(
            rf" & $\infty$ & \multicolumn{{2}}{{c}}{{$-\sqrt{{\pi/2}}={SHEET_LIMIT_EXACT:.9f}$}} & & "
            f"{SHEET_LIMIT_SCAN:.6f} & {SHEET_LIMIT_SCAN:.6f} & {SHEET_LIMIT_EXACT:.6f} & {SHEET_LIMIT_EXACT:.6f}" + " \\\\",
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_dfa_rows(rows: list[dict[str, float | str]]) -> str:
    """Main-text energy table rows with the r2SCAN column added (6-dp, as Table tab:energies)."""
    lines = ["% generated by codes/10_fci_gaussian_validation.py: Geometry & lam & Exact & SCAN & r2SCAN & "
             "DC-SCAN & DC-r2SCAN & DC-SCAN - exact"]
    for row in rows:
        geo = 'Isotropic' if row['lambda'] == 1.0 else str(row['family']).capitalize()
        lam = f"$10^{{{int(round(math.log10(row['lambda'])))}}}$" if row['lambda'] >= 1000 else f"${row['lambda']:.0f}$"
        vals = [row['ex_closed'], row['ex_scan'], row['ex_r2scan'], row['ex_dcscan'], row['ex_dcr2scan'],
                row['ex_dcscan'] - row['ex_closed']]
        cells = [f"${v:+.6f}$" if i == 5 else f"${v:.6f}$" for i, v in enumerate(vals)]
        lines.append(f"{geo}&{lam}&" + "&".join(cells) + r"\\")
    return "\n".join(lines) + "\n"


def latex_rows(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "% generated by codes/10_fci_gaussian_validation.py",
        r"\begin{tabular}{llrrrrrr}",
        r"family & $\lambda$ & $E_x^{\rm FCI}$ & Eq.~(7) & $E_{\rm FCI}$ & $J$ & $\Delta E_x$ & $\Delta J$ " + "\\\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['family']} & {row['lambda']:.0f} & {row['ex_fci']:.9f} & "
            f"{row['ex_eq7']:.9f} & {row['e_fci']:.9f} & {row['j_fci']:.9f} & "
            f"{row['ex_fci_minus_closed']:+.2e} & {row['j_fci_minus_eq7']:+.2e}"
            + " \\\\",
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows: list[dict[str, float | str]] = []
    for family in ("sheet", "wire"):
        for lam in LAMBDAS:
            omega = trap_omega(family, lam)
            ex_eq7, eq7_error = exact_eq7(omega)
            j = -ex_eq7
            h00, t00, v_harm, e_fci, ex_fci = fci_one_orbital(omega, j)
            ex_closed = exact_closed_form(family, lam)
            dfa = dfa_exchange(family, lam)
            rows.append({
                "family": family,
                "lambda": lam,
                "omega_x": omega[0],
                "omega_y": omega[1],
                "omega_z": omega[2],
                "h00": h00,
                "t00": t00,
                "v_harm": v_harm,
                "e_fci": e_fci,
                "j_fci": e_fci - 2.0 * h00,
                "ex_fci": ex_fci,
                "ex_eq7": ex_eq7,
                "ex_closed": ex_closed,
                "eq7_quad_error": eq7_error,
                "ex_eq7_minus_closed": ex_eq7 - ex_closed,
                "ex_fci_minus_closed": ex_fci - ex_closed,
                "j_fci_minus_eq7": (e_fci - 2.0 * h00) - (-ex_eq7),
                "ex_lda": dfa['LDA'],
                "ex_scan": dfa['SCAN'],
                "ex_r2scan": dfa['r2SCAN'],
                "ex_scan_closed_form": dfa['SCAN_closed_form'],
                "ex_dcscan": dfa['DC-SCAN'],
                "ex_dcr2scan": dfa['DC-r2SCAN'],
                "ex_scan_direct": dfa['SCAN_direct'],
                "ex_r2scan_direct": dfa['r2SCAN_direct'],
                "r2scan_minus_scan": dfa['r2SCAN'] - dfa['SCAN'],
                "libxc_zero_points": dfa['libxc_zero_points'],
            })

    max_ref_error = max(abs(float(r["ex_eq7_minus_closed"])) for r in rows)
    max_fci_error = max(abs(float(r["j_fci_minus_eq7"])) for r in rows)
    # At lambda=10^4, E_FCI is O(10^8 Ha); subtracting 2*h00 from the
    # double-precision FCI total leaves a harmless ~10^-9 Ha cancellation
    # error.  The independent Eq. (7) quadrature and closed form remain
    # checked to much tighter precision.
    if max_ref_error > 5.0e-11 or max_fci_error > 5.0e-9:
        raise RuntimeError(f"FCI validation failed: max reference error={max_ref_error:g}, max FCI error={max_fci_error:g}")

    csv_path = RESULTS / "fci_gaussian_validation.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for family in ("sheet", "wire"):
        fam_rows = [r for r in rows if r["family"] == family]
        (RESULTS / f"tab_fci_{family}_rows.tex").write_text(latex_rows(fam_rows))
        (RESULTS / f"tab_fci_{family}_exchange_rows.tex").write_text(latex_exchange_rows(fam_rows))
        (RESULTS / f"tab_r2scan_{family}_rows.tex").write_text(latex_dfa_rows(fam_rows))

    max_r2 = max(abs(float(r["r2scan_minus_scan"])) for r in rows)
    max_cf = max(abs(float(r["ex_scan"]) - float(r["ex_scan_closed_form"])) for r in rows)
    max_dc = max(abs(float(r["ex_dcr2scan"]) - float(r["ex_dcscan"])) for r in rows)
    max_dir = max(max(abs(float(r["ex_scan"]) - float(r["ex_scan_direct"])),
                      abs(float(r["ex_r2scan"]) - float(r["ex_r2scan_direct"]))) for r in rows)
    print("SCAN / r2SCAN exchange (Libxc) on the fixed densities")
    print("family  lambda        exact          SCAN        r2SCAN       DC-SCAN     DC-r2SCAN   r2SCAN-SCAN")
    for row in rows:
        print(f"{row['family']:6s} {row['lambda']:6.0f} {row['ex_closed']:12.6f} {row['ex_scan']:13.6f} "
              f"{row['ex_r2scan']:13.6f} {row['ex_dcscan']:12.6f} {row['ex_dcr2scan']:12.6f} "
              f"{row['r2scan_minus_scan']:+.2e}")
    print(f"sheet lam->inf: exact {SHEET_LIMIT_EXACT:.9f}, SCAN = r2SCAN {SHEET_LIMIT_SCAN:.6f}")
    print(f"max |r2SCAN - SCAN| (Libxc)            = {max_r2:.3e} Ha")
    print(f"max |SCAN Libxc - closed-form alpha=0| = {max_cf:.3e} Ha")
    print(f"max |DC-r2SCAN - DC-SCAN|              = {max_dc:.3e} Ha")
    print(f"max |F_x route - direct Libxc sum|     = {max_dir:.3e} Ha")

    print("FCI Gaussian validation: one spatial orbital, two-electron singlet")
    print("The FCI total is E_FCI = 2 h00 + J; exchange is E_x = -J.")
    print("family  lambda       E_x(FCI)       Eq.(7)       E_FCI          J       dE_x")
    for row in rows:
        print(
            f"{row['family']:6s} {row['lambda']:6.0f} "
            f"{row['ex_fci']:14.9f} {row['ex_eq7']:14.9f} "
            f"{row['e_fci']:14.9f} {row['j_fci']:12.9f} "
            f"{row['ex_fci_minus_closed']:+.2e}"
        )
    print(f"max |Eq.(7) exchange - closed form| = {max_ref_error:.3e} Ha")
    print(f"max |FCI J - Eq.(7) J|              = {max_fci_error:.3e} Ha")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
