"""Apply the jcp_main insight (one orbital -> tau=tau_W -> alpha=0) to the scalar Q2D functional
of 2D_Ref/main.pdf, and cross-check both papers on their common IBM well."""
import sys, numpy as np
np.seterr(all='ignore')
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q2d_scalar as q
from pyscf.dft import libxc

def ex_libxc(code, n, g2, tau):
    rho = np.stack((n, np.sqrt(g2), 0*n, 0*n, tau))
    return n * libxc.eval_xc(code, rho, spin=0, deriv=0)[0]

def weight(n, g2, tau, new=False):            # independent re-implementation of Eq.(4) (+ W_new factor)
    K = np.cbrt(3*np.pi**2); ctf = .3*K**2
    a = (tau - g2/(8*n)) / (ctf*n**(5/3)); p = g2/(4*K**2*n**(8/3))
    t = np.clip((a-.6)/.4, 0, 1); S = 1-10*t**3+15*t**4-6*t**5
    u = a*p; chi = p/np.sqrt(np.abs(a)+1e-6)
    W = S*u**2/(u**2+1e-10)*chi**6/(chi**6+6**6)
    if new: W = W*p**12/(p**12+6.0**12)
    act = (n > 1e-60) & (g2 > 0) & (a > 0) & (a < 1)
    return np.where(act, np.nan_to_num(W), 0.0), a

def loggrid(m, lo=1e-7, hi=12.0):
    x, w = np.polynomial.legendre.leggauss(m)
    y = np.log(lo) + (x+1)/2*(np.log(hi)-np.log(lo))
    return np.exp(y), w*(np.log(hi)-np.log(lo))/2*np.exp(y)

def gaussian(geom, lam, m=1500):
    R, wR = loggrid(m); T, wT = loggrid(m)
    R, T = np.meshgrid(R, T, indexing="ij"); w = np.outer(wR, wT)
    if geom == "sheet":   # (x,y)=R, z=T/lam
        n = 2*lam/np.pi**1.5*np.exp(-R**2-T**2); g2 = 4*n**2*(R**2+lam**2*T**2); vol = 2*np.pi*R*2/lam
    else:                 # rho=R/lam, z=T
        n = 2*lam**2/np.pi**1.5*np.exp(-R**2-T**2); g2 = 4*n**2*(lam**2*R**2+T**2); vol = 2*np.pi*R*2/lam**2
    tau = g2/(8*n)        # one doubly occupied real orbital: tau = tau_W exactly
    m_ = n > 1e-300; n, g2, tau, w = n[m_], g2[m_], tau[m_], (w*vol)[m_]
    out = {"N": w@n,
           "SCAN": w@ex_libxc("MGGA_X_SCAN", n, g2, tau),
           "r2SCAN": w@ex_libxc("MGGA_X_R2SCAN", n, g2, tau),
           "Q2D_orig(code)": w@q.evaluate(n, g2, tau).exchange}
    W, a = weight(n, g2, tau); Wn, _ = weight(n, g2, tau, True)
    out["max|alpha|"] = np.abs(a).max(); out["max W_orig"] = W.max(); out["max W_new"] = Wn.max()
    # forced W=1 (pure Q2D exchange branch) on the same density
    k = n > 1e-60
    out["pure Q2D (W=1)"] = w[k]@np.nan_to_num(q._q2d(n[k], g2[k], tau[k] + 1e-12*q.CTF*n[k]**(5/3))[1].value)
    return out

jcp = {("isotropic",1):(-0.797885,-0.795058,-0.795058),("sheet",10):(-1.179303,-1.672690,-1.082995),
       ("sheet",100):(-1.245397,-3.179886,-1.184779),("sheet",1e4):(-1.253234,-7.045312,-1.238506),
       ("wire",10):(-2.400278,-3.649731,-2.126863),("wire",100):(-4.227637,-16.362299,-4.193878),
       ("wire",1e4):(-7.901840,-292.064785,-7.990438)}
print("PART A  Gaussian families of jcp_main Eq.(densities)  [Ha]")
print(f"{'geom':9s}{'lam':>7s}{'N':>10s}{'exact':>11s}{'SCAN(jcp)':>12s}{'SCAN(here)':>12s}{'r2SCAN':>12s}{'Q2Dscal':>12s}{'DC-SCAN(jcp)':>13s}{'maxW':>7s}{'maxWn':>7s}{'W=1 Ex':>9s}")
for (g, lam), (ex, sc, dc) in jcp.items():
    o = gaussian("sheet" if g != "wire" else "wire", lam)
    print(f"{g:9s}{lam:7.0e}{o['N']:10.6f}{ex:11.6f}{sc:12.6f}{o['SCAN']:12.6f}{o['r2SCAN']:12.6f}{o['Q2D_orig(code)']:12.6f}{dc:13.6f}"
          f"{o['max W_orig']:7.0e}{o['max W_new']:7.0e}{o['pure Q2D (W=1)']:9.1e}")

# ---------------- PART B: common IBM well, r_s^2D = 4 ----------------
rs = 4.0; ns = 1/(np.pi*rs**2); kF = np.sqrt(2)/rs
def FF(Q):   # analytic form factor of the sin^2 infinite well
    Q = np.asarray(Q, float); P = 4*np.pi**2
    return np.where(Q < 1e-6, 1.0, (3*Q + 2*P/Q - 2*P**2*(-np.expm1(-Q))/(Q**2*(Q**2+P)))/(Q**2+P))
def fock(L, m=4000):
    x, w = np.polynomial.legendre.leggauss(m); x = (x+1)/2; w = w/2
    return -kF*w@((1-2/np.pi*(np.arcsin(x)+x*np.sqrt(1-x**2)))*FF(2*kF*L*x))
def well(L, m=12000):
    x, w = np.polynomial.legendre.leggauss(m); z = (x+1)/2*L; w = w*L/2
    f = 2*np.sin(np.pi*z/L)**2/L; fp = 2*np.pi/L**2*np.sin(2*np.pi*z/L)
    n = ns*f; g2 = (ns*fp)**2; tau = g2/(8*n) + np.pi*ns/2*n
    k = n > 1e-300; n, g2, tau, w = n[k], g2[k], tau[k], w[k]
    W, a = weight(n, g2, tau); Wn, _ = weight(n, g2, tau, True)
    lda = -q.CX*np.cbrt(n); r2x = ex_libxc("MGGA_X_R2SCAN", n, g2, tau); scx = ex_libxc("MGGA_X_SCAN", n, g2, tau)
    _, qx, _ = q._q2d(n, g2, tau); qx = np.nan_to_num(qx.value)
    return dict(SCAN=w@scx/ns, r2SCAN=w@r2x/ns, orig=w@(r2x+W*(qx-r2x))/ns, new=w@(r2x+Wn*(qx-r2x))/ns,
                code=w@q.evaluate(n, g2, tau).exchange/ns, Wmax=W.max(), Wnmax=Wn.max(), amin=a.min())
# numerical check of the form factor
zz, ww = np.polynomial.legendre.leggauss(400); zz = (zz+1)/2; ww = ww/2; f = 2*np.sin(np.pi*zz)**2
Qt = 3.0
def _g(u):  # autocorrelation of f on [0,1]
    zz2, w2 = np.polynomial.legendre.leggauss(200); z2 = u + (zz2+1)/2*(1-u)
    return (1-u)/2*w2@(2*np.sin(np.pi*z2)**2*2*np.sin(np.pi*(z2-u))**2)
uu, wu = np.polynomial.legendre.leggauss(200); uu = (uu+1)/2; wu = wu/2
Fnum = 2*sum(wi*_g(ui)*np.exp(-Qt*ui) for ui, wi in zip(uu, wu))
print(f"\nform factor check Q=3: analytic {float(FF(Qt)):.10f}  numeric {Fnum:.10f}")
jcpibm = {15.39:(-0.0938,-0.0950,-0.0956,-0.0925,-0.0932), 1.14:(-0.1431,-0.2426,-0.2426,-0.1635,-0.1634),
          1e-2:(-0.1500,-0.7681,-0.7683,-0.1831,-0.1831), 9.6e-5:(-0.1501,-1.2905,-1.2906,-0.1869,-0.1869)}
print("\nPART B  IBM well r_s^2D=4, eps_x per electron [Ha]; jcp_main Table (tab:ibm) vs 2D_Ref functional")
print(f"{'L':>9s}{'exact jcp':>10s}{'Fock here':>11s}{'SCAN jcp':>10s}{'SCAN here':>11s}{'r2S jcp':>9s}{'r2S here':>10s}"
      f"{'DC-SCAN':>9s}{'Q2D orig':>10s}{'(code)':>10s}{'Q2D new':>10s}{'maxW':>7s}{'maxWn':>7s}")
for L, (ex, sc, r2, dc, dcr2) in jcpibm.items():
    o = well(L, 20000 if L < 0.05 else 12000)
    print(f"{L:9.2e}{ex:10.4f}{fock(L):11.5f}{sc:10.4f}{o['SCAN']:11.5f}{r2:9.4f}{o['r2SCAN']:10.5f}{dc:9.4f}"
          f"{o['orig']:10.5f}{o['code']:10.5f}{o['new']:10.5f}{o['Wmax']:7.3f}{o['Wnmax']:7.3f}")


# ---------------- PART C: Gaussian-confined gases (alpha > 0): how W responds to the last lambda ----------------
# sheet: n(z)=n_s (lam/sqrt(pi)) exp(-lam^2 z^2), uniform 2D gas in (x,y), r_s^2D=4  -> tau = tau_W + n kF^2/4
# wire : n(rho)=n_1 (lam^2/pi) exp(-lam^2 rho^2), uniform 1D gas along z, r_s^1D=1   -> tau = tau_W + n kF^2/6
# exact exchange: jcp_main Gaussian form factors F(q)=e^{q^2/2lam^2}erfc(q/sqrt2 lam), V(k)=e^x E1(x), x=k^2/2lam^2
from scipy.special import erfcx, exp1
def exact_sheet(lam, m=4000):
    x, w = np.polynomial.legendre.leggauss(m); x = (x+1)/2; w = w/2
    F = erfcx(2*kF*x/(np.sqrt(2)*lam))
    return -kF*w@((1-2/np.pi*(np.arcsin(x)+x*np.sqrt(1-x**2)))*F)
n1 = 0.5; kF1 = np.pi*n1/2
def exact_wire(lam, m=4000):
    x, w = np.polynomial.legendre.leggauss(m); qq = (x+1)*kF1; w = w*kF1
    X = qq**2/(2*lam**2); V = np.exp(X)*exp1(X)
    return -(1/(2*np.pi**2*n1))*w@((2*kF1-qq)*V)
def gas(geom, lam, m=6000):
    x, w = np.polynomial.legendre.leggauss(m)
    t = np.exp(np.log(1e-6)+(x+1)/2*(np.log(12.0)-np.log(1e-6))); wt = w*(np.log(12.0)-np.log(1e-6))/2*t
    if geom == "sheet":
        z = t/lam; n = ns*lam/np.sqrt(np.pi)*np.exp(-t**2); g2 = (2*lam**2*z*n)**2
        tau = g2/(8*n) + n*(2*np.pi*ns)/4; dv = 2*wt/lam; N = ns
    else:
        r = t/lam; n = n1*lam**2/np.pi*np.exp(-t**2); g2 = (2*lam**2*r*n)**2
        tau = g2/(8*n) + n*kF1**2/6; dv = 2*np.pi*r*wt/lam; N = n1
    k = n > 1e-60; n, g2, tau, dv = n[k], g2[k], tau[k], dv[k]
    W, a = weight(n, g2, tau); Wn, _ = weight(n, g2, tau, True)
    r2x = ex_libxc("MGGA_X_R2SCAN", n, g2, tau); scx = ex_libxc("MGGA_X_SCAN", n, g2, tau)
    qx = np.nan_to_num(q._q2d(n, g2, tau)[1].value)
    ex = exact_sheet(lam) if geom == "sheet" else exact_wire(lam)
    return dict(exact=ex, Nchk=dv@n/N, SCAN=dv@scx/N, r2SCAN=dv@r2x/N, orig=dv@(r2x+W*(qx-r2x))/N,
                code=dv@q.evaluate(n, g2, tau).exchange/N, new=dv@(r2x+Wn*(qx-r2x))/N, Q2D=dv@qx/N,
                Wbar=dv@(n*W)/N, Wnbar=dv@(n*Wn)/N, amin=a.min(), amax=a.max())
print("\nPART C  Gaussian-confined gases, eps_x per electron [Ha];  <W> = electron-averaged weight")
print(f"{'geom':6s}{'lam':>8s}{'N/N':>8s}{'exact':>10s}{'SCAN':>10s}{'r2SCAN':>10s}{'Q2D orig':>10s}{'(code)':>10s}"
      f"{'Q2D new':>10s}{'pureQ2D':>10s}{'<W>':>7s}{'<Wn>':>7s}{'a_min':>9s}{'a_max':>9s}")
for geom in ("sheet", "wire"):
    for lam in (0.1, 0.3, 1, 3, 10, 30, 100, 1e3, 1e4):
        o = gas(geom, lam)
        print(f"{geom:6s}{lam:8.0e}{o['Nchk']:8.5f}{o['exact']:10.5f}{o['SCAN']:10.4f}{o['r2SCAN']:10.4f}{o['orig']:10.5f}"
              f"{o['code']:10.5f}{o['new']:10.5f}{o['Q2D']:10.5f}{o['Wbar']:7.3f}{o['Wnbar']:7.3f}{o['amin']:9.2e}{o['amax']:9.2e}", flush=True)
print("2D limit -4sqrt2/(3 pi rs) =", -4*np.sqrt(2)/(3*np.pi*rs))
