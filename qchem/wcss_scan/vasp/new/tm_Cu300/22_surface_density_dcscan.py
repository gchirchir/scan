#!/usr/bin/env python3
"""22 -- Surface densities of metal (111) slabs and fixed-density surface XC energies incl. DC-SCAN.

Benchmark target (user 2026-11-20): the quasi-2D surface analysis of
  L. Chiodo, L. A. Constantin, E. Fabiano, F. Della Sala, PRL 108, 126402 (2012)   [Fig. 2, Table I]
  A. Patra, S. Jana, L. A. Constantin, L. Chiodo, P. Samal, JCP 152, 151101 (2020) [Fig. 2, Table I]

Protocol (standalone, PySCF PBC, GTH pseudopotentials, gth-dzvp-molopt-sr, Gaussian smearing):
  1. self-consistent PBEsol (as the PRL density analysis) for a symmetric fcc(111) slab of N layers with vacuum
     and for an M-layer periodic bulk cell with the same 1x1 in-plane cell and lattice constant;
  2. n, grad n, tau from the Kohn-Sham orbitals on the uniform FFT grid, Hessian of n by FFT;
  3. plane-averaged density difference Delta n(z) = n_slab(z) + n_slab(-z) - n_bulk(z), with z measured from the
     centre of the broken bond (half an interlayer spacing outside the first atomic plane), i.e. the PRL/JCP Fig. 2
     construction (left + right surface minus the uncut bulk); plotted against z/d0, d0 = a/sqrt(3);
  4. fixed-density XC energies E_xc^F[n] on both densities and the surface XC energy
        sigma_xc^F = (E_xc^F[slab] - (N/M) E_xc^F[bulk]) / (2 A)
     for LDA, PBE, PBEsol, Q2D-GGA (Libxc GGA_X_Q2D/GGA_C_Q2D), SCAN and DC-SCAN (DC exchange of the DC-SCAN note,
     Eqs. 23-25, on the Libxc SCAN base, + SCAN correlation).  sigma^F - sigma^PBEsol at the PBEsol density is the
     first-order (non-self-consistent) estimate of the functional change of the surface energy.
  Also reported: the self-consistent PBEsol surface energy and the density-weighted plane average of s(z).

Usage:
  python3 22_surface_density_dcscan.py run --system Al --a 4.02 --layers 8 --kxy 8 --kz 4
  python3 22_surface_density_dcscan.py report          # CSV + figure from all results/surface22_*.npz
Outputs: results/surface22_<sys>_<tag>.npz/.json, results/surface22_summary.csv, results/surface22_dn_profiles.csv,
         figures/surface22_delta_n.pdf.
Caveats: valence (pseudo) densities without core; unrelaxed slabs at the PRL PBEsol lattice constants; a single
slab thickness unless repeated with --layers.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
HA_PER_BOHR2_TO_J_PER_M2 = 4.3597447222071e-18 / (0.529177210903e-10) ** 2
ANG = 1.0 / 0.529177210903
HA2EV = 27.211386245988

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


# ---------------------------------------------------------------- geometry
def fcc111_cell(symbol, a_ang, layers, vacuum_ang, periodic_bulk=False):
    """1x1 hexagonal fcc(111) cell; ABC stacking along z.  periodic_bulk: no vacuum, layers must be a multiple of 3."""
    from pyscf.pbc import gto
    a_in = a_ang / math.sqrt(2.0)
    d0 = a_ang / math.sqrt(3.0)
    a1 = np.array([a_in, 0.0, 0.0])
    a2 = np.array([0.5 * a_in, 0.5 * math.sqrt(3.0) * a_in, 0.0])
    shift = (a1 + a2) / 3.0
    if periodic_bulk:
        if layers % 3:
            raise SystemExit("bulk cell needs a multiple of 3 layers")
        height, z0 = layers * d0, 0.5 * d0
    else:
        height, z0 = (layers - 1) * d0 + vacuum_ang, 0.5 * vacuum_ang
    atoms = [[symbol, tuple((k % 3) * shift + np.array([0.0, 0.0, z0 + k * d0]))] for k in range(layers)]
    cell = gto.Cell()
    cell.a = np.array([a1, a2, [0.0, 0.0, height]])
    cell.atom = atoms
    cell.unit = "Angstrom"
    cell.basis = "gth-dzvp-molopt-sr"
    cell.pseudo = "gth-pbe"
    cell.verbose = 3
    return cell, d0, [z0 + k * d0 for k in range(layers)]


def run_scf(cell, kmesh, sigma_ev, cutoff, conv_tol=1e-8, max_cycle=150, kerker=None):
    """PBEsol KRKS SCF with Gaussian smearing.  kerker=None: PySCF DIIS.  kerker=(alpha, q0, nhist): Kerker-
    preconditioned Pulay mixing of the grid density and its analytic gradient (rho_kerker_scf, copied verbatim from
    codes/SurfaceMetals_cutoff/cutoff_conv.py), needed for d-metal slabs, whose DIIS SCF diverges; accepted when the
    rms density residual < 1e-6 and the standard-functional orbital gradient |g|/sqrt(n) < 1e-4."""
    from pyscf.pbc import dft
    from pyscf.pbc.scf.addons import smearing_
    cell.ke_cutoff = cutoff
    cell.build()
    if cell.nelectron % 2:
        raise SystemExit(f"odd electron count {cell.nelectron}: choose an even number of layers for this metal")
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    mf = dft.KRKS(cell, kpts)
    mf.xc = "GGA_X_PBE_SOL,GGA_C_PBE_SOL"
    mf = smearing_(mf, sigma=sigma_ev / HA2EV, method="gaussian")
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    if kerker:
        alpha, q0, nhist = kerker
        conv, it, res = rho_kerker_scf(mf, mf.get_init_guess(cell, "atom"), alpha, q0, nhist, max_cycle)
        e, g = finish_pulay(mf)
        print(f"Kerker SCF: converged={conv} iterations={it} rms={res:.2e} E={e:.10f} |g|/sqrt(n)={g:.2e}", flush=True)
        mf.converged, mf.e_tot = bool(conv and g < 1e-4), e
        mf.scf_info = dict(solver="kerker-rho", alpha=alpha, q0=q0, nhist=nhist, iterations=it, rms_residual=res,
                           grad_norm_standard=g)
    else:
        e = mf.kernel()
        mf.scf_info = dict(solver="diis", iterations=None)
    if not mf.converged:
        raise SystemExit("SCF not converged")
    return mf, e


def rho_kerker_scf(mf, dm0, alpha=0.3, q0=0.8, nhist=8, max_cycle=300, tol_rho=1e-6, log=print):
    """Pulay mixing of the grid density with Kerker preconditioning (plane-wave style; Claude Code 2026-11-21).

    KRKS + FFTDF + LDA/GGA only.  The Kohn-Sham matrix is built from the mixed input density rho_in(r): Hartree from
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
    assert xctype in ("LDA", "GGA"), xctype
    deriv = 0 if xctype == "LDA" else 1
    h1e, s1e = mf.get_hcore(cell), mf.get_ovlp(cell)
    cplx = not np.allclose(kpts, 0)

    def rho_of(dm):
        """Density and (GGA) its analytic gradient on the FFT grid: array (1 or 4, ng)."""
        dms = np.asarray(dm)
        make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, 1)
        out = np.zeros((1 + 3 * deriv, ng))
        for etc, p0, p1 in mydf.aoR_loop(grids, kpts, deriv=deriv):
            out[:, p0:p1] += np.array(make_rho(0, etc[0], etc[2], xctype), ndmin=2).real[:1 + 3 * deriv]
        return out

    def fock_of(x):
        rG = tools.fft(x[0], mesh)
        vH = tools.ifft(coulG * rG, mesh).real
        rho4 = x if deriv else x[0]
        vxc = ni.eval_xc_eff(mf.xc, rho4, deriv=1, xctype=xctype)[1]
        wv = np.array(vxc, ndmin=2) * w
        wv[0] += vH * w
        wv[0] *= 0.5
        nk, nao = len(kpts), cell.nao_nr()
        V = np.zeros((nk, nao, nao), dtype=np.complex128 if cplx else float)
        for etc, p0, p1 in mydf.aoR_loop(grids, kpts, deriv=deriv):
            for k in range(nk):
                ao = np.asarray(etc[0][k])
                ao = ao if ao.ndim == 3 else ao[None]
                aow = np.einsum("xgi,xg->gi", ao[:wv.shape[0]], wv[:, p0:p1])
                V[k] += lib.dot(ao[0].conj().T, aow)
        V = V + V.conj().transpose(0, 2, 1)
        return np.asarray(h1e) + V

    if np.asarray(dm0).ndim != 3:
        raise SystemExit("rho_kerker_scf supports KRKS (even electron count) only")
    x_in = rho_of(dm0)
    filt = lambda r: np.array([tools.ifft(P * tools.fft(c, mesh), mesh).real for c in r])
    hist_x, hist_r = [], []
    conv, rnorm = False, None
    for it in range(1, max_cycle + 1):
        F = fock_of(x_in)
        mo_e, mo_c = mf.eig(F, s1e)
        mo_occ = mf.get_occ(mo_e, mo_c)
        mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_e, mo_c, mo_occ
        R = rho_of(mf.make_rdm1(mo_c, mo_occ)) - x_in
        rnorm = float(np.sqrt(np.sum(R[0] * R[0]) * w / cell.vol))
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
                B[i, j] = B[j, i] = float(np.dot(hist_r[i][0], hist_r[j][0]))
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


FUNCS = {"LDA": ("LDA_X", "LDA_C_PW"), "PBE": ("GGA_X_PBE", "GGA_C_PBE"),
         "PBEsol": ("GGA_X_PBE_SOL", "GGA_C_PBE_SOL"), "Q2D-GGA": ("GGA_X_Q2D", "GGA_C_Q2D"),
         "SCAN": ("MGGA_X_SCAN", "MGGA_C_SCAN")}


def xc_energies(f, w):
    out = {}
    n = f["n"]
    ex_lda = eps("LDA_X", f)
    for name, (x, c) in FUNCS.items():
        ex, ec = eps(x, f), eps(c, f)
        out[name] = (float(np.sum(n * ex) * w), float(np.sum(n * ec) * w))
        if name == "SCAN":
            with np.errstate(divide="ignore", invalid="ignore"):
                fs = np.where(f["ok"], ex / np.where(f["ok"], ex_lda, 1.0), 1.0)
            ex_dc = fx_dcscan(f["s"], f["D"], f["Z"], fs, "full") * ex_lda
            out["DC-SCAN"] = (float(np.sum(n * ex_dc) * w), out["SCAN"][1])
    return out


def plane_profile(F, f):
    mesh = F["mesh"]
    n3 = F["rho"].reshape(mesh)
    nz = n3.mean(axis=(0, 1))
    s3 = np.where(f["ok"], f["s"], 0.0).reshape(mesh)
    nzw = np.where(nz > 0, nz, 1.0)
    s_av = (n3 * s3).mean(axis=(0, 1)) / nzw              # density-weighted plane average of s
    D3 = np.where(f["ok"], f["D"], 3.0).reshape(mesh)
    D_av = (n3 * D3).mean(axis=(0, 1)) / nzw
    return nz, s_av, D_av


def cmd_run(args):
    os.makedirs(RES, exist_ok=True)
    tag = f"{args.system}_L{args.layers}_k{args.kxy}_c{int(args.cutoff)}"
    out = {"system": args.system, "a_ang": args.a, "layers": args.layers, "vacuum_ang": args.vacuum,
           "kxy": args.kxy, "kz_bulk": args.kz, "cutoff_ha": args.cutoff, "sigma_ev": args.sigma,
           "density_functional": "PBEsol", "basis": "gth-dzvp-molopt-sr", "pseudo": "gth-pbe"}
    t0 = time.time()
    res = {}
    for kind, layers, kmesh in (("bulk", args.bulk_layers, [args.kxy, args.kxy, args.kz]),
                                ("slab", args.layers, [args.kxy, args.kxy, 1])):
        cell, d0, zpl = fcc111_cell(args.system, args.a, layers, args.vacuum, periodic_bulk=(kind == "bulk"))
        kk = (args.kerker_alpha, args.kerker_q0, args.kerker_hist) if (args.kerker_alpha and kind == "slab") else None
        mf, e = run_scf(cell, kmesh, args.sigma, args.cutoff, max_cycle=args.max_cycle, kerker=kk)
        F = fields_on_grid(mf)
        f = descriptors(F)
        gerr = float(np.max(np.abs(F["grad"] - F["gradF"])) / max(np.max(np.abs(F["grad"])), 1e-30))
        nz, s_av, D_av = plane_profile(F, f)
        zgrid = (np.arange(F["mesh"][2]) / F["mesh"][2]) * cell.lattice_vectors()[2, 2] / ANG   # Angstrom
        E = xc_energies(f, F["w"])
        res[kind] = dict(layers=layers, energy=float(e), nelectron=int(cell.nelectron),
                         nelec_grid=F["nelec_grid"], grad_fft_relerr=gerr, mesh=[int(m) for m in F["mesh"]],
                         area_bohr2=float(np.linalg.norm(np.cross(cell.lattice_vectors()[0], cell.lattice_vectors()[1]))),
                         xc=E, zplanes_ang=zpl, d0_ang=d0, scf=mf.scf_info)
        np.savez(os.path.join(RES, f"surface22_{tag}_{kind}.npz"), z_ang=zgrid, n_z=nz, s_z=s_av, D_z=D_av,
                 zplanes_ang=np.array(zpl), d0_ang=d0)
        print(f"{kind}: E={e:.8f} Ha, N_grid={F['nelec_grid']:.6f}/{cell.nelectron}, FFT-grad rel.err {gerr:.2e}")
    A = res["slab"]["area_bohr2"]
    ratio = res["slab"]["layers"] / res["bulk"]["layers"]
    sig = {"scf_PBEsol": (res["slab"]["energy"] - ratio * res["bulk"]["energy"]) / (2 * A) * HA_PER_BOHR2_TO_J_PER_M2}
    for name in res["slab"]["xc"]:
        xs, cs = res["slab"]["xc"][name]
        xb, cb = res["bulk"]["xc"][name]
        sig[f"x_{name}"] = (xs - ratio * xb) / (2 * A) * HA_PER_BOHR2_TO_J_PER_M2
        sig[f"xc_{name}"] = (xs + cs - ratio * (xb + cb)) / (2 * A) * HA_PER_BOHR2_TO_J_PER_M2
    out.update(res=res, sigma_J_m2=sig, seconds=time.time() - t0)
    with open(os.path.join(RES, f"surface22_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(sig, indent=1))


def delta_n(tag):
    """PRL/JCP Fig. 2 construction from the stored plane profiles of one run."""
    b = np.load(os.path.join(RES, f"surface22_{tag}_bulk.npz"))
    s = np.load(os.path.join(RES, f"surface22_{tag}_slab.npz"))
    d0 = float(s["d0_ang"])
    zc = float(s["zplanes_ang"][0]) - 0.5 * d0             # broken-bond centre outside the first plane
    zs, ns = s["z_ang"], s["n_z"]
    period = zs[1] - zs[0] + zs[-1]
    zb, nb = b["z_ang"], b["n_z"]
    pb = zb[1] - zb[0] + zb[-1]
    zb0 = float(b["zplanes_ang"][0]) - 0.5 * d0              # a bond centre of the bulk cell (layer A below)
    area = json.load(open(os.path.join(RES, f"surface22_{tag}.json")))["res"]["slab"]["area_bohr2"]
    x = np.linspace(-2.5, 1.0, 351)                          # z/d0 as in PRL/JCP Fig. 2: < 0 into the slab
    z = -x * d0                                              # distance into the slab from the broken-bond centre

    def per(zq, zz, nn, p):
        return np.interp(np.mod(zq, p), np.append(zz, p), np.append(nn, nn[0]))

    n_r = per(zc + z, zs, ns, period)                        # right (this) surface
    n_l = per(zc - z, zs, ns, period)                        # mirror image = left surface
    # bulk profile phase: layer k of the slab at zc + (k+1/2) d0 matches the bulk layer of the same stacking letter
    n_b = per(zb0 + z, zb, nb, pb)
    # n(z) = int dx dy n(x,y,z) per surface cell (papers' definition): plane average x cell area (bohr^2)
    n_r, n_l, n_b = area * n_r, area * n_l, area * n_b
    return x, n_r + n_l - n_b, n_r, n_b, np.interp(np.mod(zc + z, period), np.append(zs, period),
                                                    np.append(s["s_z"], s["s_z"][0]))


def ref_prl():
    import csv
    fn = os.path.join(HERE, "reference_prl2012_surface_energies.csv")
    return {r["system"]: {k: float(v) for k, v in r.items() if k != "system"}
            for r in csv.DictReader(l for l in open(fn) if not l.startswith("#"))}


def latex_block(rows):
    """Manuscript block (red, generated): figure + table of fixed-density surface XC shifts vs PRL Table I."""
    ref = ref_prl()
    names = ("LDA", "PBE", "Q2D-GGA", "SCAN", "DC-SCAN")
    full_lines = []
    shift_lines = []
    by_system = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)
    systems = list(ref) + [s for s in by_system if s not in ref]
    for system in systems:
        calculated = by_system.get(system, [])
        if not calculated:
            pending = f"{system}(111), PySCF pending"
            full_lines.append(pending + " & " + " & ".join(["--"] * 6) + r" \\")
            shift_lines.append(pending + " & " + " & ".join(["--"] * 6) + r" \\")
        for r in calculated:
            tag = f"{r['system']}(111), {r['layers']}L, ${r['kxy']}\\times{r['kxy']}$, {int(r['cutoff_ha'])}\\,Ha"
            sigma0 = r["sigma_scf_PBEsol"]
            full_lines.append(tag + f" & {sigma0:.2f} & " +
                              " & ".join(f"{sigma0 + r[f'dxc_{n}_minus_PBEsol']:.2f}" for n in names) + r" \\")
            shift_lines.append(tag + " & 0.00 & " +
                               " & ".join(f"{r[f'dxc_{n}_minus_PBEsol']:+.2f}" for n in names) + r" \\")
        if system in ref:
            p = ref[system]
            source = rf"\quad {system} PRL Table I (self-consistent)"
            full_lines.append(source + f" & {p['PBEsol']:.2f} & " +
                              " & ".join(f"{p[n]:.2f}" if n in p else "--" for n in names) + r" \\")
            shift_lines.append(source + " & 0.00 & " +
                               " & ".join(f"{p[n] - p['PBEsol']:+.2f}" if n in p else "--" for n in names) + r" \\")
    full_body = "\n".join(full_lines)
    shift_body = "\n".join(shift_lines)
    return r"""% generated by codes/22_surface_density_dcscan.py report --tex -- do not edit between the markers
\begin{figure}[t]
\centering
\includegraphics[width=0.62\textwidth]{figures/surface22_delta_n.pdf}
\caption{\added{Plane-integrated surface density difference $\Delta n(z)=n_{\rm s,l}+n_{\rm s,r}-n_{\rm bulk}$ (top) and
density-weighted plane average of the reduced gradient $s$ (bottom) for the (111) slabs of Table~\ref{tab:surface22},
PBEsol densities, $z$ in units of the interlayer spacing $d_0$ from the centre of the broken bond (surface atom at
$-0.5$), as in Fig.~2 of Refs.~\onlinecite{Chiodo2012,Patra2020}. The dotted line marks $s=1.76$, above which the
Q2D-GGA exchange is de-enhanced.}}
\label{fig:surface22}
\end{figure}
\begin{table*}[t]
\centering
\caption{\added{Surface energies of metal (111) slabs (J\,m$^{-2}$). Panel (a) gives the full values before any
difference is taken: the PBEsol column is the self-consistent surface energy and the other calculated columns are
the first-order estimates $\sigma^F=\sigma^{\rm PBEsol}+\Delta\sigma^F$, all evaluated on the same PBEsol density.
Panel (b) gives the corresponding differences $\Delta\sigma^F=\sigma^F-\sigma^{\rm PBEsol}$. DC-SCAN denotes DC
exchange on the SCAN base plus SCAN correlation. Rows ``PRL Table I'' are the self-consistent values of
Ref.~\onlinecite{Chiodo2012} (relaxed slabs, each functional at its own lattice constant). GTH pseudopotentials and
valence densities are used for the present calculations. ``PySCF pending'' rows and dashes identify calculations
that have not completed and are not used in any comparison.}}
\label{tab:surface22}
\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Slab/source & PBEsol & LDA & PBE & Q2D-GGA & SCAN & DC-SCAN \\
\midrule
\multicolumn{7}{@{}l}{\textit{(a) Full surface energies, $\sigma^F$}} \\
""" + full_body + r"""
\addlinespace
\multicolumn{7}{@{}l}{\textit{(b) Difference from PBEsol, $\Delta\sigma^F$}} \\
""" + shift_body + r"""
\bottomrule
\end{tabular}
\end{table*}"""


PLANNED_CUTOFF = {"Ti": 250}          # Ha; cutoff scan (SurfaceMetals_cutoff, job 6389342); others 150 Ha


def table2_rows(rows):
    """Row blocks of Table 2 of surface22_table.tex (status and protocol columns), panels (a) and (b).

    Calculated rows from results/surface22_*.json (label 'production' for 8 layers and 8x8, else 'pilot'); one
    literature row per PRL Table I metal; metals without a calculation get a 'pending' row (planned 8 layers and
    cutoff).
    """
    ref = ref_prl()
    names = ("PBEsol", "LDA", "PBE", "Q2D-GGA", "SCAN", "DC-SCAN")
    by_system = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)
    a_lines, b_lines = [], []
    for system in list(ref) + [s for s in by_system if s not in ref]:
        calc = sorted(by_system.get(system, []), key=lambda r: (r["layers"], r["kxy"], r["cutoff_ha"]))
        if not calc:
            pend = (f"{system}(111), PySCF production & \\added{{pending}} & 8 & -- & -- & "
                    f"{PLANNED_CUTOFF.get(system, 150)} & " + " & ".join(["--"] * 6) + r" \\")
            a_lines.append(pend); b_lines.append(pend)
        for r in calc:
            kind = "production" if (r["layers"] == 8 and r["kxy"] == 8) else "pilot"
            kz = r.get("kz_bulk", r["kxy"])
            proto = (f"{system}(111), PySCF {kind} & complete & {r['layers']} & ${r['kxy']}\\times{r['kxy']}\\times1$ & "
                     f"${r['kxy']}\\times{r['kxy']}\\times{kz}$ & {int(r['cutoff_ha'])}")
            s0 = r["sigma_scf_PBEsol"]
            full = [s0] + [s0 + r[f"dxc_{n}_minus_PBEsol"] for n in names[1:]]
            a_lines.append(proto + " & " + " & ".join(f"{v:.2f}" for v in full) + r" \\")
            b_lines.append(proto + " & 0.00 & " + " & ".join(f"{r[f'dxc_{n}_minus_PBEsol']:+.2f}" for n in names[1:])
                           + r" \\")
        if system in ref:
            p = ref[system]
            src = rf"\quad {system} PRL Table I & literature SCF & -- & -- & -- & --"
            a_lines.append(src + " & " + " & ".join(f"{p[n]:.2f}" if n in p else "--" for n in names) + r" \\")
            b_lines.append(src + " & 0.00 & " + " & ".join(f"{p[n] - p['PBEsol']:+.2f}" if n in p else "--"
                                                            for n in names[1:]) + r" \\")
    hdr = "% generated by 22_surface_density_dcscan.py report (table2_rows) -- do not edit by hand\n"
    for tag, lines in (("a", a_lines), ("b", b_lines)):
        with open(os.path.join(RES, f"surface22_table2_{tag}.tex"), "w") as fh:
            fh.write(hdr + "\n".join(lines) + "\n")


def write_tex(rows, tex):
    import shutil
    block = latex_block(rows)
    with open(os.path.join(RES, "surface22_jcp_block.tex"), "w") as fh:   # not surface22_table.tex (= the report)
        fh.write(block + "\n")
    if not tex:
        return
    figdir = os.path.join(os.path.dirname(os.path.abspath(tex)), "figures")
    os.makedirs(figdir, exist_ok=True)
    shutil.copy(os.path.join(FIG, "surface22_delta_n.pdf"), os.path.join(figdir, "surface22_delta_n.pdf"))
    t = open(tex).read()
    a, b = t.index("%BEGIN surface22\n"), t.index("%END surface22")
    t = t[:a] + "%BEGIN surface22\n" + block + "\n" + t[b:]
    open(tex, "w").write(t)
    print(f"updated {tex} and {figdir}/surface22_delta_n.pdf")


def cmd_report(_args):
    import csv
    files = sorted(glob.glob(os.path.join(RES, "surface22_*.json")))
    if not files:
        raise SystemExit("no results/surface22_*.json yet")
    rows, prof = [], {}
    for fn in files:
        d = json.load(open(fn))
        tag = os.path.basename(fn)[len("surface22_"):-5]
        sig = d["sigma_J_m2"]
        row = {"tag": tag, "system": d["system"], "layers": d["layers"], "kxy": d["kxy"], "cutoff_ha": d["cutoff_ha"],
               "kz_bulk": d.get("kz_bulk", d["kxy"]),
               "sigma_scf_PBEsol": sig["scf_PBEsol"]}
        for k, v in sig.items():
            if k.startswith(("x_", "xc_")):
                row[k] = v
        for name in ("LDA", "PBE", "Q2D-GGA", "SCAN", "DC-SCAN"):
            row[f"dxc_{name}_minus_PBEsol"] = sig[f"xc_{name}"] - sig["xc_PBEsol"]
        rows.append(row)
        prof[tag] = delta_n(tag)
    with open(os.path.join(RES, "surface22_summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(RES, "surface22_dn_profiles.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tag", "z_over_d0", "delta_n", "n_surface", "n_bulk", "s_planeavg"])
        for tag, (x, dn, nr, nb, sz) in prof.items():
            for i in range(len(x)):
                wr.writerow([tag, f"{x[i]:.4f}", f"{dn[i]:.6e}", f"{nr[i]:.6e}", f"{nb[i]:.6e}", f"{sz[i]:.4f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
    for tag, (x, dn, nr, nb, sz) in prof.items():
        ax[0].plot(x, dn, label=tag.split("_")[0] + "(111)")
        ax[1].plot(x, sz, label=tag.split("_")[0] + "(111)")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel(r"$\Delta n(z)=n_{\rm s,l}+n_{\rm s,r}-n_{\rm bulk}$ (a.u.)")
    ax[0].legend()
    ax[1].axhline(1.76, color="grey", ls=":", lw=0.8)
    ax[1].set_ylabel(r"$\langle s\rangle_n(z)$")
    ax[1].set_xlabel(r"$z/d_0$ (0 = broken-bond centre, $<0$ into the slab, surface atom at $-0.5$)")
    ax[1].set_ylim(0, 4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "surface22_delta_n.pdf"))
    table2_rows(rows)
    write_tex(rows, _args.tex)
    print(open(os.path.join(RES, "surface22_summary.csv")).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True)
    r.add_argument("--a", type=float, required=True, help="fcc lattice constant (Angstrom)")
    r.add_argument("--layers", type=int, default=8)
    r.add_argument("--bulk-layers", type=int, default=3)
    r.add_argument("--vacuum", type=float, default=14.0, help="Angstrom")
    r.add_argument("--kxy", type=int, default=8)
    r.add_argument("--kz", type=int, default=4, help="k points along z for the bulk cell")
    r.add_argument("--cutoff", type=float, default=30.0, help="kinetic-energy cutoff (Ha)")
    r.add_argument("--sigma", type=float, default=0.05, help="Gaussian smearing (eV)")
    r.add_argument("--kerker-alpha", type=float, default=0.0, help="slab SCF by Kerker density mixing (0 = PySCF DIIS)")
    r.add_argument("--kerker-q0", type=float, default=0.8, help="Kerker wave vector (bohr^-1)")
    r.add_argument("--kerker-hist", type=int, default=12, help="Pulay history length")
    r.add_argument("--max-cycle", type=int, default=150)
    rp = sub.add_parser("report")
    rp.add_argument("--tex", default=None, help="manuscript .tex with %%BEGIN surface22 / %%END surface22 markers")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
