import json
import os
import pytangram as tg


MAX_Q = 10  # max qubits per queue
RESULTS_FILE = "results.json"


def make_transfer_event(name, ni, ni1, sig):
    def ev(self, state):
        if state[ni] <= 0:
            return tg.DISABLED
        return state.set(**{ni: state[ni] - 1, ni1: state[ni1] + 1})
    ev.__name__ = name
    return tg.event(dist=tg.EXP(lambda s, _sig=sig: _sig))(ev)


def make_service_event(name, ni, mu):
    def ev(self, state):
        if state[ni] <= 0:
            return tg.DISABLED
        return state.set(**{ni: state[ni] - 1})
    ev.__name__ = name
    return tg.event(dist=tg.EXP(lambda s, _mu=mu: _mu))(ev)


def make_reward_func(name, ev_name):
    def ev(self, state):
        return 1.0
    ev.__name__ = name
    return tg.impulse_reward(on_event=ev_name)(ev)


def build_model(K, lam, mu, sigmas):
    """Build a K-phase model with given parameters."""

    # --- State variables ---
    state_vars = {f"n{i}": tg.var(0, max=MAX_Q) for i in range(1, K + 1)}

    # --- Events ---
    events = {}

    # n1++ (arrival)
    @tg.event(dist=tg.EXP(lambda s: lam))
    def ev_arrival(self, state):
        if state["n1"] >= MAX_Q:
            return tg.DISABLED
        return state.set(n1=state["n1"] + 1)
    events["ev_arrival"] = ev_arrival

    # Service events: one per queue (n_i--) at rate mu
    for i in range(1, K + 1):
        name = f"ev_mu_{i}"
        events[name] = make_service_event(name, f"n{i}", mu)

    # Transfer events: n_i--, n_{i+1}++ for i = 1..K-1
    for i in range(1, K):
        name = f"ev_transfer_{i}"
        events[name] = make_transfer_event(name, f"n{i}", f"n{i+1}", sigmas[i - 1])

    # Final decay: n_K-- (discarded)
    nk = f"n{K}"
    sig_k = sigmas[K - 1]

    @tg.event(dist=tg.EXP(lambda s, _sig=sig_k: _sig))
    def ev_decay(self, state):
        if state[nk] <= 0:
            return tg.DISABLED
        return state.set(**{nk: state[nk] - 1})
    events["ev_decay"] = ev_decay

    # --- Impulse rewards ---
    rewards = {}

    # Service rewards (one per queue)
    for i in range(1, K + 1):
        name = f"count_mu_{i}"
        rewards[name] = make_reward_func(name, f"ev_mu_{i}")

    for i in range(1, K):
        name = f"count_transfer_{i}"
        rewards[name] = make_reward_func(name, f"ev_transfer_{i}")

    @tg.impulse_reward(on_event="ev_decay")
    def count_decay(self, state):
        return 1.0
    rewards["count_decay"] = count_decay

    # --- Assemble class ---
    cls_dict = {**state_vars, **events, **rewards, "__module__": __name__}
    return type(f"PhaseModel{K}", (tg.Object,), cls_dict)


if __name__ == "__main__":
    K = int(input("K (number of phases): "))
    LAMBDA = float(input("Lambda (arrival rate): "))
    MU = float(input("Mu (service rate): "))
    SIGMAS = []
    for i in range(1, K + 1):
        SIGMAS.append(float(input(f"Sigma{i} (phase {i} decay rate): ")))

    print()
    ModelClass = build_model(K, LAMBDA, MU, SIGMAS)
    model = tg.Model(ModelClass())
    sol = model.solve()

    # --- Collect results ---
    obj_name = ModelClass.__name__
    result = {
        "K": K,
        "Lambda": LAMBDA,
        "Mu": MU,
        "Sigmas": SIGMAS,
        "MAX_Q": MAX_Q,
        "States": sol.space.n_states,
        "Transitions": sol.space.n_transitions,
        "Packet served": round(sum(sol[f"{obj_name}.count_mu_{i}"] for i in range(1, K + 1)), 6),
        "Discarded": round(sol[f"{obj_name}.count_decay"], 6),
    }
    for i in range(1, K):
        result[f"Decayed n{i}->n{i+1}"] = round(
            sol[f"{obj_name}.count_transfer_{i}"], 6
        )

    # --- Append to JSON ---
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            history = json.load(f)
    else:
        history = []
    history.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)

    # --- Pretty print ---
    labels = {
        "Packet served": "",
        "Discarded": "",
    }
    for i in range(1, K):
        labels[f"Decayed n{i}->n{i+1}"] = ""

    print(f"K:          {K}")
    print(f"States:     {sol.space.n_states}")
    print(f"Transitions:{sol.space.n_transitions}")
    print()
    print(f"  {'Metric':<25} {'Rate (/s)':>12}")
    for key in result:
        if key.startswith("Decayed") or key in ("Packet served", "Discarded"):
            print(f"  {key + ':':<25} {result[key]:>10.4f}")

    print(f"\nSaved to {RESULTS_FILE}")
