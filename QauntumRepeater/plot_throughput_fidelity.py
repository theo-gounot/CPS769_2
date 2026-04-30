"""Plot throughput vs fidelity threshold for Erlang quantum queue.

Fidelity threshold is modelled as the decay rate at the final phase.
Higher threshold (stricter) => higher decay rate => lower throughput.

Transfer rate between phases is kept fixed and independent.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytangram as tg

MAX_Q = 3


def make_service(name, ni, mu):
    def ev(self, state):
        if state[ni] <= 0:
            return tg.DISABLED
        return state.set(**{ni: state[ni] - 1})
    ev.__name__ = name
    return tg.event(dist=tg.EXP(lambda s, _mu=mu: _mu))(ev)


def make_transfer(name, ni, nj, tr):
    def ev(self, state):
        if state[ni] <= 0:
            return tg.DISABLED
        return state.set(**{ni: state[ni] - 1, nj: state[nj] + 1})
    ev.__name__ = name
    return tg.event(dist=tg.EXP(lambda s, _tr=tr: _tr))(ev)


def make_decay(name, nk, dr):
    def ev(self, state):
        if state[nk] <= 0:
            return tg.DISABLED
        return state.set(**{nk: state[nk] - 1})
    ev.__name__ = name
    return tg.event(dist=tg.EXP(lambda s, _dr=dr: _dr))(ev)


def make_reward(name, on_event):
    def ev(self, state):
        return 1.0
    ev.__name__ = name
    return tg.impulse_reward(on_event=on_event)(ev)


def build_model(K, lam, mu, transfer_rate, decay_rate):
    state_vars = {f"n{i}": tg.var(0, max=MAX_Q) for i in range(1, K + 1)}
    events, rewards = {}, {}

    # Arrival
    @tg.event(dist=tg.EXP(lambda s: lam))
    def ev_arrival(self, state):
        if state["n1"] >= MAX_Q:
            return tg.DISABLED
        return state.set(n1=state["n1"] + 1)
    events["ev_arrival"] = ev_arrival

    # Service (one per phase)
    for i in range(1, K + 1):
        events[f"ev_mu_{i}"] = make_service(f"ev_mu_{i}", f"n{i}", mu)

    # Transfer n_i -> n_{i+1} (fixed rate, independent of decay)
    for i in range(1, K):
        events[f"ev_transfer_{i}"] = make_transfer(f"ev_transfer_{i}", f"n{i}", f"n{i+1}", transfer_rate)

    # Decay from n_K (decay_rate = fidelity threshold)
    events["ev_decay"] = make_decay("ev_decay", f"n{K}", decay_rate)

    # Rewards
    for i in range(1, K + 1):
        rewards[f"count_mu_{i}"] = make_reward(f"count_mu_{i}", f"ev_mu_{i}")
    for i in range(1, K):
        rewards[f"count_transfer_{i}"] = make_reward(f"count_transfer_{i}", f"ev_transfer_{i}")
    rewards["count_decay"] = make_reward("count_decay", "ev_decay")

    cls_dict = {**state_vars, **events, **rewards, "__module__": __name__}
    return type(f"PhaseModel{K}", (tg.Object,), cls_dict)


# --- Sweep ---
LAMBDA        = 1.0
MU            = 1.0
TRANSFER_RATE = 1.0

# Decay rate = fidelity threshold proxy (higher = stricter)
DECAY_RATES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
K_VALUES    = [1, 2, 3, 4, 5]

for K in K_VALUES:
    fidelities  = []
    throughputs = []
    for d in DECAY_RATES:
        ModelClass = build_model(K, LAMBDA, MU, TRANSFER_RATE, d)
        model = tg.Model(ModelClass())
        sol = model.solve()
        obj = ModelClass.__name__
        served = sum(sol[f"{obj}.count_mu_{i}"] for i in range(1, K + 1))
        decay  = sol[f"{obj}.count_decay"]
        fidelity = served / (served + decay) if (served + decay) > 0 else 0.0
        fidelities.append(fidelity)
        throughputs.append(served)

    print(f"\nK={K}  (states={sol.space.n_states}):")
    for d, f, t in zip(DECAY_RATES, fidelities, throughputs):
        print(f"  decay={d:>5.2f}  fidelity={f:.4f}  throughput={t:.4f}")

    plt.plot(fidelities, throughputs, "-o", label=f"K={K}", linewidth=1.3, markersize=4)

plt.xlabel("Fidelity (achieved)", fontsize=12)
plt.ylabel("Throughput (packets/s)", fontsize=12)
plt.title(f"Throughput vs Fidelity\n(λ={LAMBDA}, μ={MU}, transfer={TRANSFER_RATE}, MAX_Q={MAX_Q})", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("throughput_vs_fidelity.png", dpi=150)
print("\nSaved: throughput_vs_fidelity.png")
