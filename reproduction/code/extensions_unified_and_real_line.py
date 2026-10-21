#!/usr/bin/env python3
"""Checks and figures for Sections 6.1, 6.2 and 7 (unified hyperbolic picture; Cauchy family on the line).

Produces figures/fig8_unified_bimodal.png, figures/fig9_real_line_cauchy.png and extension_checks.json.
Runtime about two minutes.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
out = {}

# ---------------------------------------------------------------- Möbius pushforward and hyperbolic harmonicity on S^2
def mobius(x, a):
    xa = x @ a
    return ((1 - a @ a) * x + np.outer(2 + 2 * xa, a)) / (1 + 2 * xa + a @ a)[:, None]
nt, nph = 400, 800
th = (np.arange(nt) + 0.5) * np.pi / nt; ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
TH, PH = np.meshgrid(th, ph, indexing="ij")
X = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
W = (np.sin(TH) * (np.pi / nt) * (2 * np.pi / nph)).reshape(-1) / (4 * np.pi)
J3 = lambda a, f: W @ f(mobius(X, a))
k_d = lambda r, d: hyp2f1(1, 1 - d / 2, 1 + d / 2, r * r) / hyp2f1(1, 1 - d / 2, 1 + d / 2, 1.0)
b = np.array([1.0, 0, 0])
out["linear_quadrature_vs_hypergeometric_d3"] = [abs(J3(r * b, lambda x: x @ b) - k_d(r, 3) * r) for r in (0.3, 0.7, 0.95)]
f_test = lambda x: x[:, 0] + 0.6 * x[:, 1] ** 2 - 0.3 * x[:, 2] * x[:, 0]
def hyp_harm_residual(a, h=2e-3, d=3):
    J0 = J3(a, f_test); grad = np.zeros(3); lap = 0.0
    for i in range(3):
        e = np.zeros(3); e[i] = h
        Jp, Jm = J3(a + e, f_test), J3(a - e, f_test)
        grad[i] = (Jp - Jm) / (2 * h); lap += (Jp - 2 * J0 + Jm) / h ** 2
    return lap + 2 * (d - 2) * (a @ grad) / (1 - a @ a), lap
out["hyperbolic_harmonic_residuals_d3"] = [hyp_harm_residual(a) for a in (np.array([0.3, 0.2, -0.1]), np.array([-0.5, 0.4, 0.3]), np.array([0.1, -0.6, 0.5]))]

# ---------------------------------------------------------------- radial functions h_{d,k} and boundary constants k(k+d-2)/(d-3)
def h_dk(d, k, r_end=1 - 1e-9):
    if d == 2:
        return (lambda r: np.asarray(r) ** k), (lambda r: k * np.asarray(r) ** (k - 1))
    r0 = 1e-4
    rhs = lambda r, y: [y[1], -((d - 1) / r + 2 * (d - 2) * r / (1 - r * r)) * y[1] + k * (k + d - 2) * y[0] / r ** 2]
    s = solve_ivp(rhs, [r0, r_end], [r0 ** k, k * r0 ** (k - 1)], rtol=1e-12, atol=1e-15, dense_output=True, method="DOP853")
    hend = s.sol(r_end)[0]
    return (lambda r: s.sol(np.clip(r, r0, r_end))[0] / hend), (lambda r: s.sol(np.clip(r, r0, r_end))[1] / hend)
bc = {}
for d, k in ((4, 1), (4, 2), (5, 2), (6, 3)):
    h, hp = h_dk(d, k)
    bc[f"d={d},k={k}"] = {"ratios_at_0.999_0.9999": [float(hp(r) / (1 - r)) for r in (0.999, 0.9999)], "predicted": k * (k + d - 2) / (d - 3)}
for k in (1, 2, 3):
    h, hp = h_dk(3, k)
    bc[f"d=3,k={k}"] = {"log_ratios": [float(hp(r) / ((1 - r) * np.log(1 / (1 - r)))) for r in (0.9999, 0.99999, 0.999999)], "predicted": k * (k + 1)}
h32, _ = h_dk(3, 2)
bc["d=3,k=2 quadrature vs ODE"] = [float(abs(J3(r * b, lambda x: 3 * x[:, 0] ** 2 - 1) / 2 - h32(r))) for r in (0.4, 0.8)]
out["boundary_constants"] = bc

# ---------------------------------------------------------------- Figure 8: bimodal fitness, d = 2, 3, 4
c = 0.3
def reduced_flow(d):
    h1, h1p = h_dk(d, 1); h2, h2p = h_dk(d, 2)
    def rhs(tau, y):
        r, p = y
        return [h2p(r) * (np.cos(p) ** 2 - 1 / d) + c * h1p(r) * np.cos(p),
                (-2 * h2(r) * np.cos(p) * np.sin(p) - c * h1(r) * np.sin(p)) / r ** 2]
    return rhs
def hit(tau, y): return y[0] - (1 - 1e-6)
hit.terminal = True; hit.direction = 1
phis0 = np.linspace(0.2, np.pi - 0.2, 11); r0 = 0.8
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
thb = np.linspace(0, np.pi, 400); fb = np.cos(thb) ** 2 + c * np.cos(thb)
ends = {}
for ax, d, tmax in zip(axes, (2, 3, 4), (6, 40, 40)):
    rhs = reduced_flow(d)
    for i in range(len(thb) - 1):
        ax.plot(np.cos(thb[i:i + 2]), np.sin(thb[i:i + 2]), color=plt.cm.viridis((fb[i] - fb.min()) / (fb.max() - fb.min())), lw=5, solid_capstyle="butt")
    ax.plot([-1, 1], [0, 0], color="0.85", lw=0.8)
    ends[d] = []
    for p0 in phis0:
        s = solve_ivp(rhs, [0, tmax], [r0, p0], events=hit, rtol=1e-10, atol=1e-12, max_step=0.05)
        r, p = s.y
        ax.plot(r * np.cos(p), r * np.sin(p), color="#1f5fa8", lw=1.2)
        ax.plot(r[0] * np.cos(p[0]), r[0] * np.sin(p[0]), "o", color="k", ms=2.5)
        ax.plot(r[-1] * np.cos(p[-1]), r[-1] * np.sin(p[-1]), "o", color="#1f5fa8", ms=5, markeredgecolor="white", markeredgewidth=0.5, zorder=6)
        ends[d].append({"phi0": float(p0), "tau_end": float(s.t[-1]), "phi_end": float(p[-1]), "f_end": float(np.cos(p[-1]) ** 2 + c * np.cos(p[-1]))})
    pmin = np.arccos(-c / 2)
    ax.plot([1], [0], marker="*", color="#c62828", ms=14, markeredgecolor="white", zorder=7)
    ax.plot([-1], [0], marker="*", color="#e07b1a", ms=11, markeredgecolor="white", zorder=7)
    ax.plot([np.cos(pmin)], [np.sin(pmin)], marker="v", color="k", ms=7, zorder=7)
    ax.set_aspect("equal"); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-0.2, 1.12); ax.axis("off")
    ax.set_title(f"$d={d}$: " + ("exits at non-critical directions" if d == 2 else "converges to critical points of $f$"), fontsize=9.5)
axes[0].text(1.0, -0.06, "global max", ha="center", va="top", fontsize=7.5, color="#c62828")
axes[0].text(-1.0, -0.06, "local max", ha="center", va="top", fontsize=7.5, color="#e07b1a")
axes[0].text(np.cos(np.arccos(-c / 2)), 1.07, "min of $f$", ha="center", fontsize=7.5)
fig.suptitle(r"Bimodal fitness $f(x)=(x\cdot b)^2+0.3\,(x\cdot b)$ on $S^{d-1}$: projected trajectories in the $(u,v)$ half-disk, boundary coloured by $f$", fontsize=9.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig8_unified_bimodal.png", dpi=200); plt.close(fig)
out["bimodal_endpoints"] = ends

# ---------------------------------------------------------------- Section 7: Cauchy family on the line
xg = np.linspace(-4000, 4000, 4_000_001); x0, g = 0.3, 1.7
p = g / (np.pi * ((xg - x0) ** 2 + g ** 2)); dx = xg[1] - xg[0]
sx = 2 * (xg - x0) / ((xg - x0) ** 2 + g ** 2); sg = 1 / g - 2 * g / ((xg - x0) ** 2 + g ** 2)
Fi = np.array([[np.sum(p * sx * sx), np.sum(p * sx * sg)], [np.sum(p * sx * sg), np.sum(p * sg * sg)]]) * dx
out["cauchy_fisher"] = {"numeric": Fi.tolist(), "expected": 1 / (2 * g * g)}
Jl = lambda x, y: (1 + y) / ((1 + y) ** 2 + x ** 2)
def rhsH(t, s_):
    x, y = s_; D = (1 + y) ** 2 + x ** 2
    return [2 * y * y * (-2 * x * (1 + y) / D ** 2), 2 * y * y * ((D - 2 * (1 + y) ** 2) / D ** 2)]
xstar = lambda C: 0.0 if abs(C) < 1e-14 else (1 - np.sqrt(1 - 4 * C * C)) / (2 * C)
starts = [(0.5, 1.0), (1.5, 0.4), (3.0, 0.8), (-2.0, 1.5), (-0.7, 2.5), (0.0, 1.2), (4.0, 2.0)]
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.4))
XX, YY = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(0, 4, 300))
ax2[0].contour(XX, YY, XX / ((1 + YY) ** 2 + XX ** 2), levels=np.linspace(-0.45, 0.45, 19), colors="0.8", linewidths=0.6)
ax2[0].contourf(XX, YY, Jl(XX, YY), levels=30, cmap="Blues", alpha=0.5)
checks = []
for (xs0, gs0) in starts:
    s = solve_ivp(rhsH, [0, 5e4], [xs0, gs0], rtol=1e-11, atol=1e-13, max_step=5.0)
    x, y = s.y; C = xs0 / ((1 + gs0) ** 2 + xs0 ** 2); pred = xstar(C)
    ax2[0].plot(x, y, color="#1f5fa8", lw=1.3); ax2[0].plot(xs0, gs0, "o", color="k", ms=3); ax2[0].plot(pred, 0, "o", color="#1f5fa8", ms=4)
    checks.append({"x0": xs0, "gamma0": gs0, "C": C, "x_end_numeric": float(x[-1]), "gamma_end": float(y[-1]), "x_star_closed_form": float(pred),
                   "ImF_drift": float(np.max(abs(x / ((1 + y) ** 2 + x ** 2) - C)))})
ax2[0].plot(0, 0, marker="*", color="#c62828", ms=14, markeredgecolor="white", zorder=5)
ax2[0].set_xlim(-5, 5); ax2[0].set_ylim(0, 4); ax2[0].set_xlabel(r"location $x_0$"); ax2[0].set_ylabel(r"scale $\gamma$")
ax2[0].set_title(r"Natural-gradient Cauchy flow under $f(x)=1/(1+x^2)$: paths are level circles of $\mathrm{Im}\,F$", fontsize=9)
xx = np.linspace(-6, 6, 600)
for g0, col in ((0.5, "#1f5fa8"), (1.0, "#e07b1a"), (2.0, "#2e7d32")):
    xs = np.array([xstar(C) for C in xx / ((1 + g0) ** 2 + xx ** 2)])
    ax2[1].plot(xx, xs ** 2 / (1 + xs ** 2), color=col, label=rf"$\gamma_0={g0}$")
ax2[1].set_xlabel(r"initial location $x_0$"); ax2[1].set_ylabel(r"limiting fitness loss $1-f(x_*)$")
ax2[1].set_title(r"Closed-form loss $x_*^2/(1+x_*^2)$, $x_*=\frac{1-\sqrt{1-4C^2}}{2C}$, $C=\frac{x_0}{(1+\gamma_0)^2+x_0^2}$", fontsize=9)
ax2[1].legend(frameon=False)
for ax in ax2: ax.spines[["top", "right"]].set_visible(False)
fig2.tight_layout(); fig2.savefig(FIG / "fig9_real_line_cauchy.png", dpi=200); plt.close(fig2)
out["real_line_checks"] = checks
tt = np.linspace(-np.pi, np.pi, 2001)[:-1]; errs = []
for a in (0.5 + 1.0j, -2.0 + 0.3j, 1.5 + 3.0j):
    w = (1j - a) / (1j + a)
    pw = (1 - abs(w) ** 2) / (2 * np.pi * abs(np.exp(1j * tt) - w) ** 2)
    x = np.tan(tt / 2); pc = a.imag / (np.pi * ((x - a.real) ** 2 + a.imag ** 2)) * 0.5 * (1 + x ** 2)
    errs.append(float(np.max(abs(pw - pc))))
out["cayley_density_error"] = errs
json.dump(out, open(HERE / "extension_checks.json", "w"), indent=1, default=float)
print("extensions: done")
