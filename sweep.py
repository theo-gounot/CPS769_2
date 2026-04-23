import json
import numpy as np
import matplotlib.pyplot as plt
import pytangram as tg


MAX_Q = 10


def make_transfer_event(name, ni, ni1, sig):
    def ev(self, state):
        if state[ni] <= 0:
            return tg.DISABLED
        return state.set(**{ni: state[ni] - 1, ni1: state[ni1] + 1})
    ev.__name__ = name
    return tg.event(dist=tg.EXP(lambda s, _ni=ni, _sig=sig: s[_ni] * _sig))(ev)


def make_reward_func(name, ev_name):
    def ev(self, state):
        return 1.0
    ev.__name__ = name
    return tg.impulse_reward(on_event=ev_name)(ev)


def build_model(K, lam, mu, sigmas):
    state_vars = {f"n{i}": tg.var(0, max=MAX_Q) for i in range(1, K + 1)}
    events = {}

    @tg.event(dist=tg.EXP(lambda s: lam))
    def ev_arrival(self, state):
        if state["n1"] >= MAX_Q:
            return tg.DISABLED
        return state.set(n1=state["n1"] + 1)
    events["ev_arrival"] = ev_arrival

    @tg.event(dist=tg.EXP(lambda s: mu))
    def ev_mu(self, state):
        if state["n1"] <= 0:
            return tg.DISABLED
        return state.set(n1=state["n1"] - 1)
    events["ev_mu"] = ev_mu

    for i in range(1, K):
        name = f"ev_transfer_{i}"
        events[name] = make_transfer_event(name, f"n{i}", f"n{i+1}", sigmas[i - 1])

    nk = f"n{K}"
    sig_k = sigmas[K - 1]

    @tg.event(dist=tg.EXP(lambda s, _nk=nk, _sig=sig_k, _mu=mu:
                          s[_nk] * _sig + (_mu if s["n1"] == 0 and _mu > 0 else 0)))
    def ev_decay(self, state):
        if state[nk] <= 0:
            return tg.DISABLED
        return state.set(**{nk: state[nk] - 1})
    events["ev_decay"] = ev_decay

    rewards = {}

    @tg.impulse_reward(on_event="ev_mu")
    def count_mu(self, state):
        return 1.0
    rewards["count_mu"] = count_mu

    for i in range(1, K):
        name = f"count_transfer_{i}"
        rewards[name] = make_reward_func(name, f"ev_transfer_{i}")

    @tg.impulse_reward(on_event="ev_decay")
    def count_decay(self, state):
        return 1.0
    rewards["count_decay"] = count_decay

    cls_dict = {**state_vars, **events, **rewards, "__module__": __name__}
    return type(f"PhaseModel{K}", (tg.Object,), cls_dict)


# --- Sweep ---
K, LAMBDA, MU = 2, 2.0, 1.0
SIGMA_VALS = np.linspace(0.05, 2.0, 40)

results = []
total = len(SIGMA_VALS) ** 2
print(f"Running {total} combinations...")
for ii, s1 in enumerate(SIGMA_VALS):
    for s2 in SIGMA_VALS:
        MC = build_model(K, LAMBDA, MU, [s1, s2])
        model = tg.Model(MC())
        sol = model.solve()
        served = sol[f"{MC.__name__}.count_mu"]
        discarded = sol[f"{MC.__name__}.count_decay"]
        results.append({
            "sigma1": round(s1, 4),
            "sigma2": round(s2, 4),
            "served": served,
            "discarded": discarded,
            "good_put": served / (served + discarded),
        })
    if (ii + 1) % 10 == 0:
        print(f"  {ii + 1}/{len(SIGMA_VALS)} rows done")

with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {len(results)} runs to sweep_results.json")


# --- Build grids ---
grid_served = np.zeros((len(SIGMA_VALS), len(SIGMA_VALS)))
grid_put = np.zeros((len(SIGMA_VALS), len(SIGMA_VALS)))

for r in results:
    i = np.argmin(np.abs(SIGMA_VALS - r["sigma1"]))
    j = np.argmin(np.abs(SIGMA_VALS - r["sigma2"]))
    grid_served[j, i] = r["served"]
    grid_put[j, i] = r["good_put"]


# --- Plot 1: Good put vs sigma1 (sigma2 as parameter) ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# Show a subset of sigma2 curves for clarity
subset_idx = np.linspace(0, len(SIGMA_VALS) - 1, 12, dtype=int)
for idx in sorted(subset_idx):
    s2 = SIGMA_VALS[idx]
    ax.plot(SIGMA_VALS, grid_put[idx, :], label=f"σ₂={s2:.2f}", linewidth=1.5)
ax.set_xlabel("σ₁ (phase 1 decay rate)", fontsize=12)
ax.set_ylabel("Good put fraction", fontsize=12)
ax.set_title(f"K={K}, λ={LAMBDA}, μ={MU}", fontsize=14)
ax.legend(fontsize=8, ncol=2, title="σ₂")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("good_put_vs_sigma1.png", dpi=150)
print("Saved good_put_vs_sigma1.png")


# --- Plot 2: Good put vs sigma2 (sigma1 as parameter) ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for idx in sorted(subset_idx):
    s1 = SIGMA_VALS[idx]
    ax.plot(SIGMA_VALS, grid_put[idx, :], label=f"σ₁={s1:.2f}", linewidth=1.5)
ax.set_xlabel("σ₂ (phase 2 decay rate)", fontsize=12)
ax.set_ylabel("Good put fraction", fontsize=12)
ax.set_title(f"K={K}, λ={LAMBDA}, μ={MU}", fontsize=14)
ax.legend(fontsize=8, ncol=2, title="σ₁")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("good_put_vs_sigma2.png", dpi=150)
print("Saved good_put_vs_sigma2.png")


# --- Plot 3: Good put contour ---
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
contour = ax.contourf(SIGMA_VALS, SIGMA_VALS, grid_put, levels=25, cmap="RdYlGn")
fig.colorbar(contour, ax=ax, label="Good put fraction")
ax.set_xlabel("σ₁")
ax.set_ylabel("σ₂")
ax.set_title(f"Good put contour (K={K}, λ={LAMBDA}, μ={MU})")
plt.tight_layout()
plt.savefig("good_put_contour.png", dpi=150)
print("Saved good_put_contour.png")
