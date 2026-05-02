#!/usr/bin/env python3
"""
Quantum repeater goodput analysis.

Compare three decay distributions (Exponential, Erlang-2, Hyperexponential)
with the same mean lifetime t* and measure goodput% vs fidelity threshold.

All three are decomposed into pure exponential events — solvable analytically
via model.solve().
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytangram as tg

# --- Parameters ---
Q = 10         # queue capacity (slots)
LAMBDA = 1.0   # arrival rate
MU = 2.0       # service rate (mu > lambda so no forced loss)
C = 0.1      # decay constant in F(t) = 1/2*(1+exp(-c*t))
RESULTS_FILE = "results.json"
PLOT_FILE = "goodput_vs_fidelity.png"

# --- Sweep range ---
FIDELITIES = np.linspace(0.505, 0.999, 100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shifted(slot_values, pos):
    """Remove element at `pos` and shift left, pad with 0s."""
    out = [v for i, v in enumerate(slot_values) if i != pos]
    while len(out) < Q:
        out.append(0)
    return out


def slot_values(state):
    """Extract slot_0 .. slot_{Q-1} as a list."""
    return [state[f"slot_{i}"] for i in range(Q)]


# ---------------------------------------------------------------------------
# Exponential decay model
# ---------------------------------------------------------------------------

def build_exponential(t_star):
    rate = 1.0 / t_star  # decay rate

    state_vars = {f"slot_{i}": tg.var(0, max=1) for i in range(Q)}
    events = {}
    rewards = {}

    # Arrival
    @tg.event(dist=tg.EXP(lambda s: LAMBDA))
    def ev_arrival(self, state):
        for i in range(Q):
            if state[f"slot_{i}"] == 0:
                upd = {f"slot_{i}": 1}
                return state.set(**upd)
        return tg.DISABLED
    events["ev_arrival"] = ev_arrival

    # Service (head of queue)
    @tg.event(dist=tg.EXP(lambda s: MU))
    def ev_service(self, state):
        if state["slot_0"] <= 0:
            return tg.DISABLED
        out = shifted(slot_values(state), 0)
        return state.set(**{f"slot_{i}": v for i, v in enumerate(out)})
    events["ev_service"] = ev_service

    # Decay events (one per slot)
    for i in range(Q):
        def make_decay(_i, _rate):
            @tg.event(dist=tg.EXP(lambda s, _r=_rate: _r))
            def ev(self, state):
                if state[f"slot_{_i}"] != 1:
                    return tg.DISABLED
                out = shifted(slot_values(state), _i)
                return state.set(**{f"slot_{j}": v for j, v in enumerate(out)})
            ev.__name__ = f"ev_decay_{_i}"
            return ev
        events[f"ev_decay_{i}"] = make_decay(i, rate)

    # Rewards
    @tg.impulse_reward(on_event="ev_service")
    def r_goodput(self, state):
        return 1.0
    rewards["r_goodput"] = r_goodput

    cls_dict = {**state_vars, **events, **rewards, "__module__": __name__}
    return type(f"ExpModel_t{t_star:.4f}", (tg.Object,), cls_dict)


# ---------------------------------------------------------------------------
# Erlang-2 decay model
# ---------------------------------------------------------------------------

def build_erlang2(t_star):
    rate = 2.0 / t_star  # each phase: rate = K/t* so sum has mean t*

    state_vars = {f"slot_{i}": tg.var(0, max=2) for i in range(Q)}
    events = {}
    rewards = {}

    # Arrival (enters phase 1)
    @tg.event(dist=tg.EXP(lambda s: LAMBDA))
    def ev_arrival(self, state):
        for i in range(Q):
            if state[f"slot_{i}"] == 0:
                upd = {f"slot_{i}": 1}
                return state.set(**upd)
        return tg.DISABLED
    events["ev_arrival"] = ev_arrival

    # Service (head of queue)
    @tg.event(dist=tg.EXP(lambda s: MU))
    def ev_service(self, state):
        if state["slot_0"] <= 0:
            return tg.DISABLED
        out = shifted(slot_values(state), 0)
        return state.set(**{f"slot_{i}": v for i, v in enumerate(out)})
    events["ev_service"] = ev_service

    # Phase advance (1 -> 2), one per slot
    for i in range(Q):
        def make_advance(_i, _rate):
            @tg.event(dist=tg.EXP(lambda s, _r=_rate: _r))
            def ev(self, state):
                if state[f"slot_{_i}"] != 1:
                    return tg.DISABLED
                return state.set(**{f"slot_{_i}": 2})
            ev.__name__ = f"ev_adv_{_i}"
            return ev
        events[f"ev_adv_{i}"] = make_advance(i, rate)

    # Decay (2 -> 0, shift), one per slot
    for i in range(Q):
        def make_decay(_i, _rate):
            @tg.event(dist=tg.EXP(lambda s, _r=_rate: _r))
            def ev(self, state):
                if state[f"slot_{_i}"] != 2:
                    return tg.DISABLED
                out = shifted(slot_values(state), _i)
                return state.set(**{f"slot_{j}": v for j, v in enumerate(out)})
            ev.__name__ = f"ev_decay_{_i}"
            return ev
        events[f"ev_decay_{i}"] = make_decay(i, rate)

    # Rewards
    @tg.impulse_reward(on_event="ev_service")
    def r_goodput(self, state):
        return 1.0
    rewards["r_goodput"] = r_goodput

    cls_dict = {**state_vars, **events, **rewards, "__module__": __name__}
    return type(f"E2Model_t{t_star:.4f}", (tg.Object,), cls_dict)


# ---------------------------------------------------------------------------
# Hyperexponential decay model
# ---------------------------------------------------------------------------

# Branch rates: p=0.5, mean = t*
# p*(1/r1) + (1-p)*(1/r2) = t*
# 0.5/10 + 0.5/r2 = 1  (in units of t*)
# r1 = 10/t*, r2 = 1/(0.5*t*) = 2/t*
# Check: 0.5*(t*/10) + 0.5*(t*/2) = t*(0.05+0.25) = 0.3*t* != t*
# Fix: 0.5/a + 0.5/b = 1 => b = 2/(2-a)
# a=10 => b = 2/(-8) = negative. Bad.
# Try a=4: b = 2/(-2) still negative.
# Need a < 2 for p=0.5. a=1.5: b = 2/0.5 = 4
# Check: 0.5/1.5 + 0.5/4 = 1/3 + 1/8 = 11/24 != 1. No.
# The equation is 0.5/r1_norm + 0.5/r2_norm = 1 where r_i = r_i_norm / t*
# So 0.5/r1n + 0.5/r2n = 1 => r2n = 2/(2-r1n)
# r1n=4: r2n = 2/(-2) = -1. No.
# r1n=1.5: r2n = 2/0.5 = 4. Check: 0.5/1.5 + 0.5/4 = 1/3 + 1/8 = 0.458. No.
# Wait: 0.5*(1/1.5) + 0.5*(1/4) = 0.5*0.667 + 0.5*0.25 = 0.333+0.125 = 0.458.
# That's not 1. The constraint is on mean, not sum-of-reciprocals.
# E[T] = p/r1 + (1-p)/r2 = 0.5*t*/r1n + 0.5*t*/r2n = t*
# => 0.5/r1n + 0.5/r2n = 1
# r1n=4: 0.5/4 + 0.5/r2n = 1 => 0.125 + 0.5/r2n = 1 => r2n = 0.5/0.875 = 4/7
# r1n=10: 0.5/10 + 0.5/r2n = 1 => r2n = 0.5/0.95 = 10/19

def build_hyper(t_star):
    # p=0.5, mean = t*
    # Constraint: 0.5/r1n + 0.5/r2n = 1 (in normalized units)
    # Choose r1n=10 (fast): r2n = 0.5/(1-0.5/10) = 10/19 (slow)
    # E[T] = 0.5*(t*/10) + 0.5*(19*t*/10) = t*. Correct.
    r1 = 10.0 / t_star
    r2 = 10.0 / (19.0 * t_star)

    state_vars = {f"slot_{i}": tg.var(0, max=2) for i in range(Q)}
    events = {}
    rewards = {}

    # Arrival with branching (50/50)
    @tg.event(dist=tg.EXP(lambda s: LAMBDA))
    def ev_arrival(self, state):
        for i in range(Q):
            if state[f"slot_{i}"] == 0:
                return tg.branches([
                    (0.5, state.set(**{f"slot_{i}": 1})),  # fast branch
                    (0.5, state.set(**{f"slot_{i}": 2})),  # slow branch
                ])
        return tg.DISABLED
    events["ev_arrival"] = ev_arrival

    # Service (head of queue)
    @tg.event(dist=tg.EXP(lambda s: MU))
    def ev_service(self, state):
        if state["slot_0"] <= 0:
            return tg.DISABLED
        out = shifted(slot_values(state), 0)
        return state.set(**{f"slot_{i}": v for i, v in enumerate(out)})
    events["ev_service"] = ev_service

    # Decay events for each slot
    for i in range(Q):
        def make_decay_fast(_i, _rate):
            @tg.event(dist=tg.EXP(lambda s, _r=_rate: _r))
            def ev(self, state):
                if state[f"slot_{_i}"] != 1:
                    return tg.DISABLED
                out = shifted(slot_values(state), _i)
                return state.set(**{f"slot_{j}": v for j, v in enumerate(out)})
            ev.__name__ = f"ev_decay_f_{_i}"
            return ev
        events[f"ev_decay_f_{i}"] = make_decay_fast(i, r1)

        def make_decay_slow(_i, _rate):
            @tg.event(dist=tg.EXP(lambda s, _r=_rate: _r))
            def ev(self, state):
                if state[f"slot_{_i}"] != 2:
                    return tg.DISABLED
                out = shifted(slot_values(state), _i)
                return state.set(**{f"slot_{j}": v for j, v in enumerate(out)})
            ev.__name__ = f"ev_decay_s_{_i}"
            return ev
        events[f"ev_decay_s_{i}"] = make_decay_slow(i, r2)

    # Rewards
    @tg.impulse_reward(on_event="ev_service")
    def r_goodput(self, state):
        return 1.0
    rewards["r_goodput"] = r_goodput

    cls_dict = {**state_vars, **events, **rewards, "__module__": __name__}
    return type(f"HyperModel_t{t_star:.4f}", (tg.Object,), cls_dict)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_model(build_fn, t_star):
    """Build, solve, and return goodput rate."""
    ModelClass = build_fn(t_star)
    model = tg.Model(ModelClass())
    sol = model.solve()

    obj_name = ModelClass.__name__
    goodput = sol[f"{obj_name}.r_goodput"]
    n_states = sol.space.n_states
    return goodput, n_states


def run_sweep():
    results = {
        "params": {"Q": Q, "lambda": LAMBDA, "mu": MU, "c": C},
        "exponential": [],
        "erlang2": [],
        "hyper": [],
    }

    builders = [
        ("exponential", build_exponential),
        ("erlang2", build_erlang2),
        ("hyper", build_hyper),
    ]

    for idx, f in enumerate(FIDELITIES):
        t_star = -np.log(2 * f - 1) / C
        print(f"[{idx+1}/{len(FIDELITIES)}] F={f:.4f}  t*={t_star:.4f}")

        for label, build_fn in builders:
            goodput, n_states = solve_model(build_fn, t_star)
            pct = (goodput / LAMBDA) * 100
            results[label].append({
                "fidelity": round(f, 4),
                "t_star": round(float(t_star), 4),
                "goodput_rate": round(float(goodput), 6),
                "goodput_pct": round(float(pct), 4),
                "states": n_states,
            })

    return results


# CV (Coefficient of Variation)
# Exponential: CV = 1.0
# Erlang-2: CV = 1/sqrt(K) = 0.707
# Hyper: p=0.5, r1=10/t*, r2=10/(19*t*)
#   E[T] = t*, E[T^2] = 2*0.5*(t*/10)^2 + 2*0.5*(19*t*/10)^2
#        = (t*/10)^2 + (19*t*/10)^2 = t*^2 * (1 + 361)/100 = 3.62*t*^2
#   Var = E[T^2] - E[T]^2 = 2.62*t*^2, SD = 1.619*t*, CV = 1.619
CV_EXP = 1.0
CV_E2 = 0.707
CV_HYPER = 1.619

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    styles = {
        "exponential": {"label": f"Exponential (CV={CV_EXP:.2f})",
                        "color": "#e63946", "marker": "o", "ls": "-"},
        "erlang2":     {"label": f"Erlang-2 (CV={CV_E2:.2f})",
                        "color": "#2a9d8f", "marker": "s", "ls": "--"},
        "hyper":       {"label": f"Hyperexponential (CV={CV_HYPER:.2f})",
                        "color": "#264653", "marker": "^", "ls": ":"},
    }

    for label, style in styles.items():
        x = [r["fidelity"] for r in results[label]]
        y_good = [r["goodput_rate"] for r in results[label]]
        y_util = [r["fidelity"] * g for r, g in zip(results[label], y_good)]

        ax1.plot(x, y_good, label=style["label"],
                 color=style["color"], marker=style["marker"],
                 linestyle=style["ls"], markersize=3, linewidth=1.5)
        ax2.plot(x, y_util, label=style["label"],
                 color=style["color"], marker=style["marker"],
                 linestyle=style["ls"], markersize=3, linewidth=1.5)

    # Top
    ax1.set_title(f"Q={Q}, $\lambda$={LAMBDA}, $\mu$={MU}, c={C}")
    ax1.set_ylabel(r"Goodput $\Lambda_{\mathrm{good}}$")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, None)

    # Bottom
    ax2.set_xlabel("Fidelity threshold F")
    ax2.set_ylabel(r"Utility ($F \times \Lambda_{\mathrm{good}}$)")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, None)

    ax2.set_xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"\nPlot saved to {PLOT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Sweeping {len(FIDELITIES)} fidelity points...")
    print(f"  Q={Q}, λ={LAMBDA}, μ={MU}, c={C}\n")

    results = run_sweep()

    # Save JSON
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            history = json.load(f)
    else:
        history = []
    history.append(results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Plot
    plot_results(results)
