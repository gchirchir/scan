#!/usr/bin/env python3
"""Equilibrium fcc lattice constants for the PRL 108, 126402 (2012) Table I recreation  (Claude Code, 2026-11-22).

STANDALONE: numpy + pyscf only; rho_kerker_scf / finish_pulay copied verbatim from prl2012_scf.py.

For one metal M and functional F (LDA, PBE, PBEsol, Q2D-GGA): self-consistent energy of the 1-atom fcc primitive cell
at 9 lattice constants a = a_PRL(M,F) x (0.95 ... 1.05), restricted KRKS + Gaussian smearing, Kerker density mixing
(as the surface runs), GTH-PBE pseudopotentials, gth-dzvp-molopt-sr basis, the per-metal cutoff of prl2012_scf.py
(Ti 250, Cu 300, else 150 Ha).  E(V) is fitted with the third-order Birch-Murnaghan equation (linear least squares in
V^(-2/3)); a0 = (4 V0)^(1/3).  Output: results/lattice_<M>_<F>.json; --table -> results/lattice_table.{csv,tex}.

Usage: python lattice_scf.py --system Pt --functional PBEsol [--k 10 --sigma 0.1]
       python lattice_scf.py --table
"""
import argparse, csv, glob, json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
REF = os.path.join(HERE, "reference_prl2012_table1_full.csv")
HA2EV = 27.211386245988
BOHR_ANG = 0.52917721092
EV_A3_TO_GPA = 160.21766208
XC = {"LDA": "LDA_X,LDA_C_PW", "PBE": "GGA_X_PBE,GGA_C_PBE", "PBEsol": "GGA_X_PBE_SOL,GGA_C_PBE_SOL",
      "Q2D-GGA": "GGA_X_Q2D,GGA_C_Q2D", "SCAN": "MGGA_X_SCAN,MGGA_C_SCAN"}
FUNCS = ["LDA", "PBE", "PBEsol", "Q2D-GGA"]        # columns with PRL reference values
EXTRA = ["SCAN"]                                    # this work only (SCAN scan also yields DC-SCAN, see dcscan_delta)
METALS = ["Al", "Cu", "Rh", "Au", "Pt", "Pd", "Ti"]
SCALES = [0.95, 0.9625, 0.975, 0.9875, 1.00, 1.0125, 1.025, 1.0375, 1.05]   # 9 points, +-5 %


# ==================================================== copied verbatim from codes/20_trap_dc_exchange.py
# (its copy of codes/01_wire_benchmark.py: constants, m2d, A2D) and its DC-SCAN switch functions
from scipy.special import gamma

CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)        # LDA exchange constant, E_x^LDA = -CX int n^(4/3)

KF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)            # k_F = KF n^(1/3)

C_NORM = 2.0 / np.pi ** 1.5                       # n = lam^k C_NORM exp(-(u + t^2)) in reduced coordinates

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

TAU_UNIF_X = 0.3 * KF ** 2                 # tau_unif = TAU_UNIF_X n^{5/3}
# ==================================================== end of copy (20)


def fields_on_grid(mf):
    """n, grad n, tau (KS orbitals, smeared occupations) and the FFT Hessian of n on the uniform grid."""
    cell = mf.cell
    ni = mf._numint
    grids = mf.grids
    dm = mf.make_rdm1()
    nao = cell.nao_nr()
    rows = []
    for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(cell, grids, nao, deriv=1, kpts=mf.kpts):
        r = ni.eval_rho(cell, ao_k1, dm, xctype="MGGA", hermi=1, with_lapl=False)
        rows.append(np.asarray(r).real)
    R = np.concatenate(rows, axis=1)
    rho, grad, tau = R[0], R[1:4], R[-1]
    mesh = np.asarray(grids.mesh) if hasattr(grids, "mesh") else np.asarray(cell.mesh)
    if rho.size != np.prod(mesh):
        raise SystemExit("grid is not the uniform FFT mesh")
    n3 = rho.reshape(mesh)
    Gv = cell.get_Gv(mesh)                                  # reciprocal vectors in the C-order of the mesh
    nG = np.fft.fftn(n3).ravel()
    gradF = np.array([np.fft.ifftn((1j * Gv[:, i] * nG).reshape(mesh)).real.ravel() for i in range(3)])
    hess = np.empty((3, 3, rho.size))
    for i in range(3):
        for j in range(i, 3):
            h = np.fft.ifftn((-Gv[:, i] * Gv[:, j] * nG).reshape(mesh)).real.ravel()
            hess[i, j] = hess[j, i] = h
    w = cell.vol / rho.size
    return dict(rho=rho, grad=grad, tau=tau, hess=hess, gradF=gradF, mesh=mesh, w=w,
                coords=np.asarray(grids.coords), nelec_grid=float(rho.sum() * w))


def descriptors(F, nmin=1e-8):
    rho, grad, hess, tau = F["rho"], F["grad"], F["hess"], F["tau"]
    ok = rho > nmin
    nn = np.where(ok, rho, 1.0)
    C = -0.5 * (hess / nn - np.einsum("ig,jg->ijg", grad, grad) / nn ** 2)
    trc2 = np.einsum("ijg,ijg->g", C, C)
    Cg = np.einsum("ijg,jg->ig", C, grad)
    gc2g = np.einsum("ig,ig->g", Cg, Cg)
    gg = np.einsum("ig,ig->g", grad, grad)
    with np.errstate(divide="ignore", invalid="ignore"):
        Dd = np.clip(np.where(gc2g > 0, gg * trc2 / np.where(gc2g > 0, gc2g, 1.0), 3.0), 1.0, 3.0)
    kf = KF * np.cbrt(nn)
    s = np.sqrt(gg) / (2.0 * kf * nn)
    tauw = gg / (8.0 * nn)
    tau_c = np.maximum(tau, tauw)                         # tau >= tau_W (numerical floor)
    return dict(ok=ok, n=np.where(ok, rho, 0.0), g=np.where(ok, grad, 0.0), tau=np.where(ok, tau_c, 0.0), s=s, D=Dd,
                Z=np.sqrt(trc2) / kf ** 2)


def eps(code, f):
    from pyscf.dft import libxc
    n = np.where(f["ok"], f["n"], 1.0)
    if code.startswith("LDA"):
        inp = n
    elif code.startswith("GGA"):
        inp = np.vstack((n, np.where(f["ok"], f["g"], 0.0)))
    else:
        inp = np.vstack((n, np.where(f["ok"], f["g"], 0.0), np.zeros_like(n), np.where(f["ok"], f["tau"], 1.0)))
    key = code + "," if ("_X_" in code or code == "LDA_X") else "," + code
    return np.where(f["ok"], np.asarray(libxc.eval_xc(key, inp, spin=0, deriv=0)[0]), 0.0)


def dcscan_delta(mf):
    """E_x^DC-SCAN - E_x^SCAN (Ha) on the current (SCAN) density: same descriptors and switch as script 22 xc_energies
    (DC-SCAN depends on the density Hessian, so it is evaluated post-SCF on the self-consistent SCAN density)."""
    F = fields_on_grid(mf)
    f = descriptors(F)
    n, w = f["n"], F["w"]
    ex_lda, ex_scan = eps("LDA_X", f), eps("MGGA_X_SCAN", f)
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = np.where(f["ok"], ex_scan / np.where(f["ok"], ex_lda, 1.0), 1.0)
    ex_dc = fx_dcscan(f["s"], f["D"], f["Z"], fs, "full") * ex_lda
    return float(np.sum(n * (ex_dc - ex_scan)) * w)


def reference():
    rows = csv.reader(l for l in open(REF) if not l.startswith("#"))
    head = next(rows)
    out = {}
    for r in rows:
        for fn, v in zip(head[2:], r[2:]):
            out[(r[0], r[1], fn)] = float(v)
    return out


def fcc_cell(symbol, a_ang, cutoff):
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.a = 0.5 * a_ang * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    cell.atom = [[symbol, (0.0, 0.0, 0.0)]]
    cell.unit = "Angstrom"
    cell.basis = "gth-dzvp-molopt-sr"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = cutoff
    cell.verbose = 3
    cell.build()
    return cell


def rho_kerker_scf(mf, dm0, alpha=0.3, q0=0.8, nhist=8, max_cycle=300, tol_rho=1e-6, log=print):
    """Pulay mixing of the grid density with Kerker preconditioning (plane-wave style; Claude Code 2026-11-21).

    KRKS + FFTDF + LDA/GGA/meta-GGA (MGGA: tau mixed like the gradient and included in the residual metric and
    the convergence test; 2026-11-22).  The Kohn-Sham matrix is built from the mixed input density rho_in(r): Hartree from
    rho_in(G), xc from [rho_in, grad rho_in] (libxc via eval_xc_eff).  The analytic gradient of each output density
    (PySCF GGA density evaluator) is mixed with the same Pulay coefficients and Kerker filter (the filter commutes
    with the gradient), so the fixed point is exactly the standard PySCF KS solution (2026-11-22; the 2026-11-21
    version used the FFT gradient and needed a density-matrix Pulay finish, which stalls for d-metal slabs).  Output density rho_out(r) from the
    smeared occupations; residual R = rho_out - rho_in.  Pulay combination of the last nhist pairs, then
    rho_in <- rho_opt + P R_opt with P(G) = alpha G^2/(G^2+q0^2) (G = 0 has R = 0 by charge conservation).
    Returns (converged, iterations, rms residual); mf.mo_* hold the last orbitals (dm_out = mf.make_rdm1()).
    """
    from pyscf.pbc import tools
    from pyscf import lib
    cell, mydf = mf.cell, mf.with_df
    ni = mf._numint
    mesh = np.asarray(mydf.mesh)
    grids = mydf.grids
    coulG = tools.get_coulG(cell, mesh=mesh)
    Gv = cell.get_Gv(mesh)
    G2 = np.einsum("gx,gx->g", Gv, Gv)
    P = alpha * G2 / (G2 + q0 * q0)
    ng = len(coulG)
    w = cell.vol / ng
    kpts = mf.kpts
    xctype = ni._xc_type(mf.xc)
    assert xctype in ("LDA", "GGA", "MGGA"), xctype
    deriv = 0 if xctype == "LDA" else 1
    nrow = {"LDA": 1, "GGA": 4, "MGGA": 5}[xctype]
    h1e, s1e = mf.get_hcore(cell), mf.get_ovlp(cell)
    cplx = not np.allclose(kpts, 0)

    def rho_of(dm):
        """Density, (GGA) its analytic gradient, (MGGA) tau on the FFT grid: array (1, 4 or 5, ng)."""
        dms = np.asarray(dm)
        make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, 1)
        out = np.zeros((nrow, ng))
        for etc, p0, p1 in mydf.aoR_loop(grids, kpts, deriv=deriv):
            r = np.array(make_rho(0, etc[0], etc[2], xctype), ndmin=2).real
            out[:, p0:p1] += r[[0, 1, 2, 3, r.shape[0] - 1]] if nrow == 5 else r[:nrow]   # MGGA: rho, grad, tau
        return out

    def fock_of(x):
        rG = tools.fft(x[0], mesh)
        vH = tools.ifft(coulG * rG, mesh).real
        rho4 = x if deriv else x[0]
        vxc = ni.eval_xc_eff(mf.xc, rho4, deriv=1, xctype=xctype)[1]
        wv = np.array(vxc, ndmin=2) * w
        wv[0] += vH * w
        wv[0] *= 0.5
        wtau = wv[4] * 0.25 if nrow == 5 else None     # PySCF: 1/2 (tau) x 1/2 (V + V^H below)
        wv = wv[:4] if nrow == 5 else wv
        nk, nao = len(kpts), cell.nao_nr()
        V = np.zeros((nk, nao, nao), dtype=np.complex128 if cplx else float)
        for etc, p0, p1 in mydf.aoR_loop(grids, kpts, deriv=deriv):
            for k in range(nk):
                ao = np.asarray(etc[0][k])
                ao = ao if ao.ndim == 3 else ao[None]
                aow = np.einsum("xgi,xg->gi", ao[:wv.shape[0]], wv[:, p0:p1])
                V[k] += lib.dot(ao[0].conj().T, aow)
                if wtau is not None:
                    for x in (1, 2, 3):
                        V[k] += lib.dot(ao[x].conj().T, ao[x] * wtau[p0:p1, None])
        V = V + V.conj().transpose(0, 2, 1)
        return np.asarray(h1e) + V

    if np.asarray(dm0).ndim != 3:
        raise SystemExit("rho_kerker_scf supports KRKS (even electron count) only")
    x_in = rho_of(dm0)
    # Kerker filter for the density and its gradient; plain linear mixing (alpha) for tau, which is not a charge
    # and whose long-wavelength part must be updated (the Kerker filter would freeze it)
    filt = lambda r: np.array([alpha * c if i == 4 else tools.ifft(P * tools.fft(c, mesh), mesh).real
                               for i, c in enumerate(r)])
    hist_x, hist_r = [], []
    conv, rnorm = False, None
    for it in range(1, max_cycle + 1):
        F = fock_of(x_in)
        mo_e, mo_c = mf.eig(F, s1e)
        mo_occ = mf.get_occ(mo_e, mo_c)
        mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_e, mo_c, mo_occ
        R = rho_of(mf.make_rdm1(mo_c, mo_occ)) - x_in
        mrows = [0, 4] if nrow == 5 else [0]          # residual metric: density (+ tau for meta-GGAs)
        rnorm = float(np.sqrt(np.sum(R[mrows] * R[mrows]) * w / cell.vol))
        log(f"rhokerker it {it:4d} rms(rho_out-rho_in) = {rnorm:.3e}  N_in = {x_in[0].sum() * w:.6f}", flush=True)
        if rnorm < tol_rho:
            conv = True
            break
        hist_x.append(x_in.copy()); hist_r.append(R.copy())
        hist_x, hist_r = hist_x[-nhist:], hist_r[-nhist:]
        n = len(hist_r)
        B = np.empty((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                B[i, j] = B[j, i] = float(np.sum(hist_r[i][mrows] * hist_r[j][mrows]))
        B[n, :n] = B[:n, n] = -1.0
        B[n, n] = 0.0
        sc = np.max(np.abs(np.diag(B)[:n])); B[:n, :n] /= sc
        rhs = np.zeros(n + 1); rhs[n] = -1.0
        c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
        x_opt = sum(ci * x for ci, x in zip(c, hist_x))
        r_opt = sum(ci * r for ci, r in zip(c, hist_r))
        x_in = x_opt + filt(r_opt)
    return conv, it, rnorm


def finish_pulay(mf):
    """Energy and orbital gradient of the standard functional at the Pulay-converged density (no further iteration:
    for d-metal slabs a plain diagonalisation step amplifies residual differences, so PySCF's extra cycle is skipped)."""
    cell = mf.cell
    h1e, s1e = mf.get_hcore(cell), mf.get_ovlp(cell)
    dm = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    veff = mf.get_veff(cell, dm)
    e = float(mf.energy_tot(dm, h1e, veff))
    fock = mf.get_fock(h1e, s1e, veff, dm)
    g = np.asarray(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock))
    return e, float(np.linalg.norm(g) / np.sqrt(g.size))



def energy(a, args, a_ang):
    from pyscf.pbc import dft
    from pyscf.pbc.scf.addons import smearing_
    cell = fcc_cell(args.system, a_ang, args.cutoff)
    kpts = cell.make_kpts([args.k] * 3, wrap_around=True)
    mf = dft.KRKS(cell, kpts)
    mf.xc = XC[args.functional]
    mf = smearing_(mf, sigma=args.sigma / HA2EV, method="gaussian")
    conv, it, res = rho_kerker_scf(mf, mf.get_init_guess(cell, "atom"), 0.3, 0.8, 12, 200)
    e, g = finish_pulay(mf)
    ok = bool(conv and g < 1e-4)
    d = dict(a_ang=a_ang, volume_ang3=float(cell.vol) * BOHR_ANG ** 3, energy=float(e), converged=ok,
             iterations=it, rms=res, grad=g)
    if args.functional == "SCAN" and ok:
        d["energy_dcscan"] = float(e) + dcscan_delta(mf)
    return d


def birch_murnaghan(V, E):
    """E = sum_k c_k x^k, x = V^(-2/3), k = 0..3 (third-order Birch-Murnaghan); returns V0, E0, B0 (eV/A^3)."""
    x = V ** (-2.0 / 3.0)
    c = np.polyfit(x, E, 3)
    p, dp, d2p = np.poly1d(c), np.poly1d(c).deriv(), np.poly1d(c).deriv(2)
    roots = [r.real for r in dp.roots if abs(r.imag) < 1e-12 and x.min() * 0.95 < r.real < x.max() * 1.05]
    if not roots:
        raise SystemExit("no minimum inside the scanned range")
    x0 = min(roots, key=lambda r: p(r))
    V0 = x0 ** (-1.5)
    # B = V d2E/dV2 ; dE/dV = p'(x) dx/dV, dx/dV = -2/3 V^(-5/3)
    dxdV, d2xdV2 = -2.0 / 3.0 * V0 ** (-5.0 / 3.0), 10.0 / 9.0 * V0 ** (-8.0 / 3.0)
    B0 = V0 * (d2p(x0) * dxdV ** 2 + dp(x0) * d2xdV2)
    resid = float(np.sqrt(np.mean((p(x) - E) ** 2)))
    return float(V0), float(p(x0)), float(B0), resid


def cmd_run(args):
    os.makedirs(RES, exist_ok=True)
    if args.cutoff is None:
        args.cutoff = {"Ti": 250.0, "Cu": 300.0}.get(args.system, 150.0)
    fref = args.functional if args.functional in FUNCS else "PBEsol"     # no PRL SCAN a: centre on PBEsol
    a_ref = reference()[("a_ang", args.system, fref)]
    t0 = time.time()
    pts = []
    for s in SCALES:
        d = energy(s, args, a_ref * s)
        print(f"a = {d['a_ang']:.4f} A  E = {d['energy']:.10f} Ha  converged={d['converged']} it={d['iterations']}",
              flush=True)
        pts.append(d)
    if not all(p["converged"] for p in pts):
        raise SystemExit("not all points converged")
    V = np.array([p["volume_ang3"] for p in pts])
    E = np.array([p["energy"] for p in pts]) * HA2EV
    V0, E0, B0, resid = birch_murnaghan(V, E)
    a0 = (4.0 * V0) ** (1.0 / 3.0)
    lo, hi = min(p["a_ang"] for p in pts), max(p["a_ang"] for p in pts)
    edge = not (lo + 0.15 * (hi - lo) < a0 < hi - 0.15 * (hi - lo))   # minimum too close to the scanned edge
    extra = {}
    if args.functional == "SCAN":
        Ed = np.array([p["energy_dcscan"] for p in pts]) * HA2EV
        V0d, E0d, B0d, rd = birch_murnaghan(V, Ed)
        extra = dict(a0_dcscan_ang=(4.0 * V0d) ** (1.0 / 3.0), B0_dcscan_GPa=B0d * EV_A3_TO_GPA,
                     fit_rms_dcscan_meV=rd * 1000)
    out = dict(system=args.system, functional=args.functional, a0_ang=a0, B0_GPa=B0 * EV_A3_TO_GPA, **extra,
               fit_rms_meV=resid * 1000, minimum_near_edge=edge, a_prl=a_ref, a_exp=reference()[("a_ang", args.system, "Exp")],
               k=args.k, sigma_ev=args.sigma, cutoff_ha=args.cutoff, points=pts, seconds=time.time() - t0)
    with open(os.path.join(RES, f"lattice_{args.system}_{args.functional}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"RESULT {args.system} {args.functional}: a0 = {a0:.4f} A (PRL {a_ref:.2f}), B0 = {B0 * EV_A3_TO_GPA:.0f} GPa"
          + ("  WARNING: minimum near the edge of the scan" if edge else ""))


def cmd_table(_a):
    ref = reference()
    got = {}
    for f in glob.glob(os.path.join(RES, "lattice_*_*.json")):
        d = json.load(open(f))
        got[(d["system"], d["functional"])] = d
    red = lambda x: f"\\textcolor{{red}}{{{x}}}"
    with open(os.path.join(RES, "lattice_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "functional", "a0_this_work", "a_prl", "diff", "a_exp", "B0_GPa", "fit_rms_meV"])
        for m in METALS:
            for fn in FUNCS:
                d = got.get((m, fn))
                if d:
                    w.writerow([m, fn, f"{d['a0_ang']:.4f}", d["a_prl"], f"{d['a0_ang'] - d['a_prl']:+.4f}",
                                d["a_exp"], f"{d['B0_GPa']:.1f}", f"{d['fit_rms_meV']:.3f}"])
    lines = ["% generated by codes/PRL2012_scf/lattice_scf.py --table -- do not edit by hand",
             "\\begin{tabular}{@{}l" + "rr" * 4 + "rr" + "r@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{fn}}}" for fn in FUNCS) + " & SCAN & DC-SCAN & " +
             red("Exp.") + " \\\\",
             " & " + " & ".join(["This work & " + red("PRL")] * 4) + " & This work & This work & \\\\", "\\midrule"]
    err = {fn: [] for fn in FUNCS}
    for m in METALS:
        cells = []
        for fn in FUNCS:
            d = got.get((m, fn))
            cells += [(f"{d['a0_ang']:.3f}" + ("$^e$" if d.get("minimum_near_edge") else "")) if d else "--",
                      red(f"{ref[('a_ang', m, fn)]:.2f}")]
            if d:
                err[fn].append(d["a0_ang"] - ref[("a_ang", m, fn)])
        d = got.get((m, "SCAN"))
        cells += ([(f"{d['a0_ang']:.3f}" + ("$^e$" if d.get("minimum_near_edge") else "")),
                   f"{d['a0_dcscan_ang']:.3f}"] if d else ["--", "--"])
        lines.append(f"{m} & " + " & ".join(cells) + " & " + red(f"{ref[('a_ang', m, 'Exp')]:.2f}") + " \\\\")
    lines.append("\\midrule")
    lines.append("ME / MAE vs PRL & " + " & ".join(
        (f"\\multicolumn{{2}}{{c}}{{{np.mean(err[fn]):+.3f} / {np.mean(np.abs(err[fn])):.3f} ({len(err[fn])})}}"
         if err[fn] else "\\multicolumn{2}{c}{--}") for fn in FUNCS) + " & & & \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(RES, "lattice_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", action="store_true")
    p.add_argument("--system", choices=METALS)
    p.add_argument("--functional", choices=FUNCS + EXTRA)
    p.add_argument("--k", type=int, default=10, help="k x k x k Monkhorst-Pack mesh (fcc primitive cell)")
    p.add_argument("--sigma", type=float, default=0.1, help="Gaussian smearing (eV)")
    p.add_argument("--cutoff", type=float, default=None)
    a = p.parse_args()
    if a.table:
        return cmd_table(a)
    if not (a.system and a.functional):
        p.error("--system and --functional are required")
    cmd_run(a)


if __name__ == "__main__":
    main()
