#!/usr/bin/env python3
"""Checks for Lemma 2, Proposition 2 and Figure 6 of the revised manuscript.

1. Lemma 2: under the wrapped-Cauchy closure m_k = a^k, the first-moment equation of the
   replicator hierarchy equals (1-|a|^2)/2 * conj(F~'(a)), F~(a) = sum_k (c_k - i s_k) a^k / k,
   for random trigonometric-polynomial fitness functions.
2. Proposition 2: boundary constants of h_d(r) = r k_d(r) for linear fitness on S^{d-1}:
   h'(r)/(1-r) -> (d-1)/(d-3) for d >= 4, and h'(r)/((1-r) log(1/(1-r))) -> 2 for d = 3.
3. Dimension four: physical-time exponents |v| ~ t^{-1/6}, 1-J ~ t^{-1/3}; Figure 6.
4. Table 1 Watson column from kappa' = 1/2 - c m2(kappa)/Var_kappa(cos 2 theta), c = 4D or gamma.

Run:  python closure_lemma_and_rates.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import hyp2f1, i0, i1, iv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
(ROOT / "figures").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
rng = np.random.default_rng(20261019)
out = {}

# ---------------------------------------------------------------- 1. Lemma 2
def closure_velocity(c, s, a):
    """First-moment velocity of the replicator hierarchy under m_k = a^k."""
    K = len(c)
    g = (c - 1j * s) / 2
    m = lambda k: a ** k if k >= 0 else np.conj(a) ** (-k)
    fbar = sum(c[k - 1] * np.real(a ** k) + s[k - 1] * np.imag(a ** k) for k in range(1, K + 1))
    Ef = sum(g[k - 1] * m(k + 1) + np.conj(g[k - 1]) * m(1 - k) for k in range(1, K + 1))
    return Ef - a * fbar

def lemma2_prediction(c, s, a):
    K = len(c)
    return (1 - abs(a) ** 2) / 2 * np.conj(sum((c[k - 1] - 1j * s[k - 1]) * a ** (k - 1) for k in range(1, K + 1)))

worst = 0.0
for _ in range(200):
    K = int(rng.integers(1, 7))
    c, s = rng.normal(size=K), rng.normal(size=K)
    a = rng.uniform(0, 0.98) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    worst = max(worst, abs(closure_velocity(c, s, a) - lemma2_prediction(c, s, a)))
out["lemma2_max_abs_error_200_random_polys"] = worst
print("Lemma 2 max |closure - prediction| over 200 random polynomials:", worst)

# ---------------------------------------------------------------- 2. boundary constants
def k_d(r, d):
    return hyp2f1(1, 1 - d / 2, 1 + d / 2, r * r) / hyp2f1(1, 1 - d / 2, 1 + d / 2, 1.0)

def h_d(r, d):
    return r * k_d(r, d)

def hp_d(r, d, e=1e-7):
    return (h_d(r + e, d) - h_d(r - e, d)) / (2 * e)

out["boundary_constants"] = {}
for d in (4, 5, 6, 8):
    ratios = [hp_d(r, d) / (1 - r) for r in (0.99, 0.999, 0.9999)]
    out["boundary_constants"][f"d={d}"] = {"ratios": ratios, "predicted": (d - 1) / (d - 3)}
    print(f"d={d}: h'/(1-r) at r=0.99,0.999,0.9999 =", np.round(ratios, 5), " predicted", (d - 1) / (d - 3))
r3 = [hp_d(r, 3) / ((1 - r) * np.log(1 / (1 - r))) for r in (0.999, 0.9999, 0.99999)]
out["boundary_constants"]["d=3_log_ratio"] = r3
print("d=3: h'/((1-r)log(1/(1-r))) =", np.round(r3, 4), " predicted limit 2 (slow convergence)")
out["k4_polynomial_check"] = max(abs(k_d(r, 4) - (3 - r * r) / 2) for r in np.linspace(0, 0.99, 50))

# ---------------------------------------------------------------- 3. dimension four in physical time
def rhs4(tau, y):
    u, v, lt = y
    r2 = u * u + v * v
    return [1.5 - 1.5 * u * u - 0.5 * v * v, -u * v, 9 / (1 - r2) ** 2 / np.exp(lt)]

J4 = lambda u, v: (3 - (u * u + v * v)) / 2 * u
sol = solve_ivp(rhs4, [0, 9.5], [0.2, 0.7, np.log(1e-3)], rtol=1e-11, atol=1e-14, dense_output=True, max_step=0.01)
slopes = {}
for tau in (6, 8):
    u, v, lt = sol.sol(tau)
    u2, v2, lt2 = sol.sol(tau - 1)
    slopes[f"tau={tau}"] = {"t": float(np.exp(lt)),
                           "dlog|v|/dlogt": float((np.log(v) - np.log(v2)) / (lt - lt2)),
                           "dlog(1-J)/dlogt": float((np.log(1 - J4(u, v)) - np.log(1 - J4(u2, v2))) / (lt - lt2))}
out["d4_physical_time_slopes"] = slopes
print("d=4 local slopes (expect -1/6 and -1/3):", json.dumps(slopes, indent=1))
landmarks = {}
tau = np.linspace(0, 9.5, 40000)
u, v, lt = sol.sol(tau)
for T in (1e2, 1e4, 1e7, 1e10):
    i = int(np.argmin(abs(lt - np.log(T))))
    landmarks[f"t={T:.0e}"] = {"v": float(v[i]), "1-J": float(1 - J4(u[i], v[i]))}
out["d4_landmarks_from_(0.2,0.7)"] = landmarks
print("d=4 landmarks from (0.2,0.7):", json.dumps(landmarks, indent=1))

# Figure 6
inits = [(0.0, 0.7), (-0.6, 0.6), (0.2, 0.7), (0.7, 0.7)]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
S = np.linspace(-3 * np.log(10), 12 * np.log(10), 800)
for (u0, v0), col in zip(inits, colors):
    s4 = solve_ivp(rhs4, [0, 9.5], [u0, v0, np.log(1e-3)], rtol=1e-11, atol=1e-14, dense_output=True, max_step=0.02)
    tt = np.linspace(0, 9.5, 4000)
    uu, vv, ll = s4.sol(tt)
    ax[0].plot(np.log10(np.exp(ll)), J4(uu, vv), color=col, label=f"({u0}, {v0})")
    ax[1].plot(np.log10(np.exp(ll)), vv, color=col)
    cc = np.sqrt(1 - v0 * v0)  # d=2 reference: v conserved, u' = (c^2-u^2)^2/2, integrated in log t
    s2 = solve_ivp(lambda s_, y: [np.exp(s_) * (cc * cc - y[0] ** 2) ** 2 / 2], [S[0], S[-1]], [u0],
                   method="LSODA", rtol=1e-10, atol=1e-13, dense_output=True)
    ax[0].plot(S / np.log(10), s2.sol(S)[0], color=col, ls="--", lw=1)
    ax[1].plot(S / np.log(10), np.full_like(S, v0), color=col, ls="--", lw=1)
tref = np.logspace(6, 12, 50)
ax[1].plot(np.log10(tref), 0.09 * (tref / 1e6) ** (-1 / 6), color="k", ls=":", lw=1)
ax[1].text(9.3, 0.07, r"$\propto t^{-1/6}$", fontsize=9)
ax[0].set_xlabel("log$_{10}$ physical time $t$"); ax[0].set_ylabel("Projected mean fitness $J$")
ax[1].set_xlabel("log$_{10}$ physical time $t$"); ax[1].set_ylabel("Transverse parameter $v$")
ax[0].legend(frameon=False, title="$(u_0,v_0)$", fontsize=9)
ax[0].set_xlim(-3, 12); ax[1].set_xlim(-3, 12); ax[1].set_yscale("log"); ax[1].set_ylim(1e-2, 1)
ax[1].legend([Line2D([], [], color="k"), Line2D([], [], color="k", ls="--")], ["$d=4$", "$d=2$"],
             frameon=False, fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig6_dim4_physical_time.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. Watson column of Table 1
m2 = lambda k: i1(k) / i0(k)
var = lambda k: (1 + iv(2, k) / i0(k)) / 2 - m2(k) ** 2
watson = {}
for coef, label in [(0.4, "diffusion_0.1"), (1.2, "diffusion_0.3"), (2.4, "diffusion_0.6"),
                    (0.1, "reset_0.1"), (0.3, "reset_0.3"), (0.6, "reset_0.6")]:
    kap = brentq(lambda k: 0.5 - coef * m2(k) / var(k), 1e-6, 50)
    watson[label] = float(m2(kap))
out["watson_projected_m2"] = watson
print("Watson projected m2:", json.dumps(watson, indent=1))

(ROOT / "results" / "closure_lemma_and_rates.json").write_text(json.dumps(out, indent=2, default=float))
print("wrote results/closure_lemma_and_rates.json and figures/fig6_dim4_physical_time.png")
