"""Scalar Laplacian-free Q2D correction with separate XC energies and derivatives.

The independent variables are n, |grad n|^2 and conventional tau (including
the factor 1/2). The C++ libfunc2 entry points convert their doubled tau first.
The Q2D crossover is spin-unpolarized.
"""
from dataclasses import dataclass

import numpy as np

XC_CODE = "MGGA_X_R2SCAN,MGGA_C_R2SCAN"
_CHANNEL_CODES = ("MGGA_X_R2SCAN", "MGGA_C_R2SCAN")
K = float(np.cbrt(3 * np.pi**2))
CTF = 0.3 * K**2
CX = float(0.75 * np.cbrt(3 / np.pi))
# Constant of the p/sqrt(alpha) gate in the weight W.  jana_ref: 6.  Larger values switch W off in single-orbital
# molecular regions (1s cores, X-H bonds); see scan_weight_variants.py.  Override with set_gate_constant().
GATE_C = 6.0


def set_gate_constant(c):
    global GATE_C
    GATE_C = float(c)


@dataclass(frozen=True)
class XC:
    """Per-volume exchange and correlation and their first partial derivatives.

    Each derivative array has shape (3, ...) in order n, |grad n|^2, tau.
    """

    exchange: np.ndarray
    correlation: np.ndarray
    exchange_derivative: np.ndarray
    correlation_derivative: np.ndarray


@dataclass(frozen=True)
class _First:
    """Array value and its derivatives with respect to the three local inputs."""

    value: np.ndarray
    derivative: np.ndarray

    @classmethod
    def variable(cls, value, index):
        slope = np.zeros((3, value.size))
        slope[index] = 1.0
        return cls(value, slope)

    @staticmethod
    def _convert(other):
        return other if isinstance(other, _First) else _First(other, 0.0)

    def __add__(self, other):
        other = self._convert(other)
        return _First(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._convert(other)
        return _First(self.value - other.value, self.derivative - other.derivative)

    def __rsub__(self, other):
        return self._convert(other) - self

    def __mul__(self, other):
        other = self._convert(other)
        return _First(self.value * other.value,
                      self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._convert(other)
        value = self.value / other.value
        return _First(value, (self.derivative - value * other.derivative) / other.value)

    def __rtruediv__(self, other):
        return self._convert(other) / self

    def __pow__(self, exponent):
        value = self.value**exponent
        return _First(value, self.derivative * (exponent * self.value**(exponent - 1)))

    def exp(self):
        value = np.exp(self.value)
        return _First(value, self.derivative * value)

    def log(self):
        return _First(np.log(self.value), self.derivative / self.value)

    def log1p(self):
        return _First(np.log1p(self.value), self.derivative / (1 + self.value))


def _constant_where(mask, value, variable):
    return _First(np.where(mask, value, variable.value),
                  np.where(mask, 0, variable.derivative))


def _complement(t):
    v = 1 - t
    return v * v * v * (1 + 3 * t + 6 * t * t)


def _cap(factor, onset, limit):
    t = (factor - onset) / (limit - onset)
    t = _constant_where(t.value <= 0, 0, t)
    t = _constant_where(t.value >= 1, 1, t)
    return limit + (factor - limit) * _complement(t)


def _gate(ratio, power):
    # Work with the smaller ratio to keep even the sixth power bounded.
    small = ratio.value <= 1
    reciprocal = np.divide(1, ratio.value, out=np.zeros_like(ratio.value), where=~small)
    bounded = _First(np.where(small, ratio.value, reciprocal),
                     ratio.derivative * np.where(small, 1, -reciprocal**2))
    value = bounded**power
    return _First(np.where(small, value.value, 1) / (1 + value.value),
                  value.derivative * np.where(small, 1, -1) / (1 + value.value)**2)


def _q2d(density, gradient_squared, tau):
    n = _First.variable(density, 0)
    g2 = _First.variable(gradient_squared, 1)
    kinetic = _First.variable(tau, 2)
    n13 = n**(1 / 3)
    pauli = kinetic - g2 / (8 * n)
    alpha = pauli / (CTF * n * n13 * n13)
    p = g2 / (4 * K**2 * n * n * n13 * n13)
    cutoff = _complement(_constant_where(alpha.value <= 0.60, 0, (alpha - 0.60) / 0.40))
    weight = cutoff * _gate(alpha * p / 1e-5, 2) * _gate(p / (GATE_C * (alpha + 1e-6)**0.5), 6)

    s = p**0.5
    match = alpha / (alpha + 1e-4)
    coefficient = 4 * np.sqrt(2) * np.sqrt(2 * CTF) / (3 * np.pi * CX)
    alpha_power = (0.96 * (-0.5 * s**0.3).exp() * alpha.log()).exp()
    denominator = (1 + 2.7 * alpha * (0.8788 + match * s)
                   - 2.5 * alpha_power / (5 + s)**0.4
                   + 0.924 * (-10 * s).exp() * (alpha * alpha * alpha).log1p())
    fx = _cap(alpha**0.5 * (coefficient + (2.7 - coefficient) * match)
              / denominator, 1.15, 1.174)
    lda = -CX * n13

    r = (n / (2 * pauli))**0.5
    r2, r32 = r * r, r * r**0.5
    v = 1.0022 * r - 0.02069 * r32 + 0.33997 * r2 + 0.01747 * r2 * r
    inv_v = 1 / v
    regular = inv_v.value >= 1e-4
    direct = _constant_where(~regular, 1e-4, inv_v)
    series = _constant_where(regular, 0.0, inv_v)
    delta = direct.log1p() / direct - 1
    approximation = series * (-0.5 + series * (1 / 3 + series * (-0.25 + 0.2 * series)))
    delta = _First(np.where(regular, delta.value, approximation.value),
                   np.where(regular, delta.derivative, approximation.derivative))
    c2d = (0.1925 * delta + ((0.0863136 - 0.1925 * 1.0022) * r
           + 0.1925 * 0.02069 * r32 + (0.0572384 - 0.1925 * 0.33997) * r2)
           * inv_v.log1p())
    rg = _constant_where(r.value >= 100, 100, r)
    g0 = 0.5 * (1 + 0.088 * rg + 0.258 * rg * rg + 0.00037 * rg * rg * rg) * (-1.46 * rg).exp()
    dx = -2.18595 * lda * alpha * alpha / (1 + 0.190934 * p * alpha * alpha)
    ratio = dx / c2d
    raw = c2d + (1 - 2 * g0) * dx / (1 + 0.8 * ratio * ratio)
    fxc = _cap(fx + raw / lda, 2.18, 2.215)
    return weight, n * lda * fx, n * lda * (fxc - fx)


def evaluate(n, gradient_squared, tau, baseline=None, *, density_tol=0.0):
    """Return separate XC energies and derivatives for spin-unpolarized inputs.

    The baseline defaults to PySCF Libxc r2SCAN. A supplied callable takes the
    same three broadcast inputs and returns XC.
    Its energies and derivatives must be per volume. Inputs may be scalars or
    broadcastable arrays. Keep the same baseline when comparing to C++.
    """
    n, gradient_squared, tau = np.broadcast_arrays(
        np.asarray(n, dtype=float), np.asarray(gradient_squared, dtype=float),
        np.asarray(tau, dtype=float))
    if baseline is None:
        baseline = pyscf_baseline
    reference = baseline(n, gradient_squared, tau)
    n13 = np.cbrt(n)
    safe_density = np.where(n > 0, n, 1)
    pauli = tau - gradient_squared / (8 * safe_density)
    uniform = CTF * n * n13 * n13
    active = ((n > max(1e-60, 2 * density_tol)) & (gradient_squared > 0)
              & (pauli > 0) & (pauli < uniform))
    output = [np.array(value, dtype=float, order="C", copy=True) for value in
              (reference.exchange, reference.correlation,
               reference.exchange_derivative, reference.correlation_derivative)]
    if np.any(active):
        mask = active.ravel()
        weight, exchange, correlation = _q2d(n.ravel()[mask],
                                             gradient_squared.ravel()[mask], tau.ravel()[mask])
        for i, branch in enumerate((exchange, correlation)):
            value = output[i].reshape(-1)
            derivative = output[i + 2].reshape(3, -1)
            difference = branch.value - value[mask]
            derivative[:, mask] += (weight.value * (branch.derivative - derivative[:, mask])
                                    + weight.derivative * difference)
            value[mask] += weight.value * difference
    return XC(*output)


def pyscf_baseline(n, gradient_squared, tau):
    """Evaluate unpolarized r2SCAN energies and first derivatives with PySCF Libxc."""
    from pyscf.dft import libxc

    n, gradient_squared, tau = np.broadcast_arrays(n, gradient_squared, tau)
    density = n.ravel()
    zero = np.zeros_like(density)
    rho = np.stack((density, np.sqrt(gradient_squared.ravel()), zero, zero, tau.ravel()))
    energy, slope = [], []
    for code in _CHANNEL_CODES:
        exc, vxc, _, _ = libxc.eval_xc(code, rho, spin=0, deriv=1)
        vrho, vgradient_squared, _, vtau = vxc
        energy.append((density * exc).reshape(n.shape))
        slope.append(np.stack((vrho, vgradient_squared, vtau)).reshape((3,) + n.shape))
    return XC(*energy, *slope)


def one_electron(n, gradient_squared, tau, baseline=None):
    """Fully polarized one-electron endpoint with spin-scaled exchange and zero Ec."""
    n, gradient_squared, tau = np.broadcast_arrays(
        np.asarray(n, dtype=float), np.asarray(gradient_squared, dtype=float),
        np.asarray(tau, dtype=float))
    if baseline is None:
        baseline = pyscf_baseline
    doubled = baseline(2 * n, 4 * gradient_squared, 2 * tau)
    derivative = doubled.exchange_derivative.copy()
    derivative[1] *= 2
    return XC(doubled.exchange / 2, np.zeros_like(n), derivative,
              np.zeros_like(derivative))


def attach_pyscf(mf, *, density_tol=0.0):
    """Configure a molecular PySCF RKS object for ordinary scalar-Q2D SCF."""
    from pyscf.dft.numint import NumInt

    if mf.mol.spin != 0 or getattr(mf, "istype", lambda _: False)("UKS"):
        raise ValueError("The scalar Q2D PySCF adapter requires molecular RKS")

    class ScalarQ2DNumInt(NumInt):
        def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None,
                        xctype=None, verbose=None, spin=None):
            if xc_code != XC_CODE:
                return super().eval_xc_eff(xc_code, rho, deriv=deriv, omega=omega,
                                           xctype=xctype, verbose=verbose, spin=spin)
            if deriv not in (0, 1):
                raise NotImplementedError("Scalar Q2D supports energy and first derivatives")
            rho = np.asarray(rho)
            if spin not in (0, None) or rho.ndim != 2 or rho.shape[0] not in (5, 6):
                raise ValueError("The scalar Q2D adapter requires unpolarized MGGA inputs")
            n, gradient, tau = rho[0], rho[1:4], rho[-1]
            result = evaluate(n, np.sum(gradient * gradient, axis=0), tau,
                              pyscf_baseline, density_tol=density_tol)
            volume = result.exchange + result.correlation
            exc = np.divide(volume, n, out=np.zeros_like(volume), where=n > 0)
            if deriv == 0:
                return [exc, None, None, None]
            dn, dg2, dtau = result.exchange_derivative + result.correlation_derivative
            effective = np.vstack((dn, 2 * gradient * dg2, dtau))
            return [exc, effective, None, None]

    mf.xc = XC_CODE
    mf._numint = ScalarQ2DNumInt()
    return mf
