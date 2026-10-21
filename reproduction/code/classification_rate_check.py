#!/usr/bin/env python3
"""Numerical check of Theorem 2(iii) (Morse classification, rates at an anisotropic maximum).

f(x) = x.b + eps [ (x.e)^2 - (x.e')^2 ] on S^{d-1}, eps = 0.15, so that -Hess_S f at the maximum b has
eigenvalues (1-2eps, 1+2eps, 1, ...): mu_1 = 0.7, kappa = tr = d-1.
Predictions: dist(zeta,m) ~ exp(-mu_1 tau); 1-r ~ exp(-kappa tau/(d-3)) (d>=4), log(1/(1-r)) ~ exp(kappa tau) (d=3);
physical-time exponent mu_1 (d-3)/(2 kappa) for d>=4.
Integration is done in x = log(1/(1-r)) to avoid underflow; d=4 uses the closed forms h_{4,1}=(3r-r^3)/2, h_{4,2}=2r^2-r^4,
d=3 uses the radial ODE for r <= 1-1e-7 and the integrating-factor continuation beyond.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
eps = 0.15
out = {}
# ---------------- d = 4
def rhs4(tau, y):
    x = y[0]; z = y[1:5]; s = np.exp(-x); r = 1 - s; omr2 = s * (2 - s)
    h1, h2 = (3 * r - r ** 3) / 2, 2 * r * r - r ** 4
    Jr_s = (3 * (1 + r) / 2) * z[0] + eps * (4 * r * (1 + r)) * (z[1] ** 2 - z[2] ** 2)
    gS = np.array([h1, 2 * eps * h2 * z[1], -2 * eps * h2 * z[2], 0.0]); gS -= (gS @ z) * z
    return [Jr_s] + list(gS / r ** 2) + [9 / omr2 ** 2 * np.exp(-y[5])]
z0 = np.array([0.3, 0.8, 0.5, 0.4]); z0 /= np.linalg.norm(z0)
sol = solve_ivp(rhs4, [0, 40], [np.log(2.0)] + list(z0) + [np.log(1e-3)], rtol=1e-11, atol=1e-14, dense_output=True, max_step=0.02)
tang = lambda z: np.linalg.norm(z[1:]) / np.linalg.norm(z)
rows = []
for tau in (10, 20, 40):
    y, yp = sol.sol(tau), sol.sol(tau - 1)
    rows.append({"tau": tau, "dlog_tang_dtau": float(np.log(tang(y[1:5]) / tang(yp[1:5]))), "dlog_s_dtau": float(-(y[0] - yp[0])),
                 "dlog_tang_dlog_t": float(np.log(tang(y[1:5]) / tang(yp[1:5])) / (y[5] - yp[5]))})
out["d4"] = {"predicted": {"dlog_tang_dtau": -(1 - 2 * eps), "dlog_s_dtau": -3.0, "dlog_tang_dlog_t": -(1 - 2 * eps) / 6}, "measured": rows}
# ---------------- d = 3
def h_dk(d, k, r_end=1 - 1e-9):
    r0 = 1e-4
    rhs = lambda r, y: [y[1], -((d - 1) / r + 2 * (d - 2) * r / (1 - r * r)) * y[1] + k * (k + d - 2) * y[0] / r ** 2]
    s = solve_ivp(rhs, [r0, r_end], [r0 ** k, k * r0 ** (k - 1)], rtol=1e-12, atol=1e-16, dense_output=True, method="DOP853")
    hend = s.sol(r_end)[0]
    return (lambda r: s.sol(np.clip(r, r0, r_end))[0] / hend), (lambda r: s.sol(np.clip(r, r0, r_end))[1] / hend)
r1 = 1 - 1e-7
H = {k: h_dk(3, k) for k in (1, 2)}
C1 = {k: r1 ** 2 * H[k][1](r1) / (1 - r1 ** 2) for k in (1, 2)}
lam = {1: 2.0, 2: 6.0}
def hp_over_s(k, x):
    s = np.exp(-x); r = 1 - s
    return (1 + r) / r ** 2 * (C1[k] + lam[k] * 0.5 * (np.log((1 + r) * (1 - r1) / (1 + r1)) + x))
def rhs3(tau, y):
    x = y[0]; z = y[1:4]; r = 1 - np.exp(-x)
    Jr_s = hp_over_s(1, x) * z[0] + eps * hp_over_s(2, x) * (z[1] ** 2 - z[2] ** 2)
    gS = np.array([1.0, 2 * eps * z[1], -2 * eps * z[2]]); gS -= (gS @ z) * z
    return [Jr_s] + list(gS / r ** 2)
z0 = np.array([0.3, 0.8, 0.5]); z0 /= np.linalg.norm(z0)
sol3 = solve_ivp(rhs3, [0, 12], [np.log(1 / 1e-7)] + list(z0), rtol=1e-11, atol=1e-14, dense_output=True, max_step=0.01)
tang3 = lambda z: np.linalg.norm(z[1:]) / np.linalg.norm(z)
rows = []
for tau in (6, 9, 12):
    y, yp = sol3.sol(tau), sol3.sol(tau - 1)
    rows.append({"tau": tau, "dlog_tang_dtau": float(np.log(tang3(y[1:4]) / tang3(yp[1:4]))), "dlog_x_dtau": float(np.log(y[0] / yp[0]))})
out["d3"] = {"predicted": {"dlog_tang_dtau": -(1 - 2 * eps), "dlog_x_dtau": 2.0}, "measured": rows}
Path(__file__).with_name("classification_rate_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
