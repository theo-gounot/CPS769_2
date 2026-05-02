# AGENT_NOTES.md

## Session: 2026-05-01 -- Model audit (erlang.py vs notes.md)

### System Description
Quantum communication link: **peer1 -> repeater -> peer2**.
Two connection segments: **p1-r** (peer1 to repeater) and **r-p2** (repeater to peer2).
Each segment has a queue of capacity K, holding qubits that await the next hop.

### State Representation (per notes)
- `v1` = queue vector for p1-r. `v2` = queue vector for r-p2.
- `v1 = [1, 2, 0]` means: slot 1 holds a qubit in Erlang phase 1, slot 2 holds a qubit in Erlang phase 2, slot 3 is empty.
- Entries are **per-qubit service phases**, NOT aggregate phase counters.
- Each entry value is in `{0, 1, ..., K}` where 0 = empty slot, K = service complete.
- Queue is FIFO-ordered: qubit at slot 1 always arrived before slot 2, etc.
- When slot 1 is served and moves to the next link, all remaining qubits shift left (qubit2 becomes qubit1, opening a new slot at the end).

### Decay Model (per notes)
- Fidelity: `F(t) = 1/2 * (1 + exp(-c*t))`
- Threshold `t*` derived from fidelity requirement. `mu = 1/t*`.
- Decay is **probabilistic and per-qubit**: each qubit starts its own decay timer on arrival.
- A qubit in an earlier Erlang phase is NOT necessarily younger -- decay cannot be "grouped" by phase.
- Bell pair fidelity inherits decay from the original qubit.
- When a qubit's elapsed time exceeds `t*`, it is discarded (or the pair is).

### Key Parameters (per notes)
- `K` = Erlang-K phase count AND queue capacity.
- `mu = 1/t*` = service rate (inverse of fidelity threshold time).
- `lambda1` = arrival rate at queue 1 (p1-r).
- `lambda2` = arrival rate at queue 2 (r-p2).
- `v1, v2` are the two queue vectors forming the double-queue system.

### Critical Errors in erlang.py
1. **State representation is wrong.** Code uses `n1, n2, ..., nK` as aggregate counters (how many qubits are in phase i). Notes use per-qubit phase vectors `v1, v2` with fixed-size slots. These are mathematically different -- the code collapses qubit identity; the notes preserve it.
2. **Decay conflated with transfer.** Code uses `ev_transfer_i` (n_i -> n_{i+1}) and `ev_decay` (n_K discard) at rates from `sigmas`. Notes say decay is per-qubit, time-based, independent of service phase. The code's "transfer" is phase progression; the notes' transfer is physical handoff between p1-r and r-p2.
3. **Missing double-queue topology.** Code has one queue chain. Notes specify two queues (v1 for p1-r, v2 for r-p2) with two arrival rates (lambda1, lambda2).
4. **No left-shift semantics.** Notes say when qubit at slot 1 departs, remaining qubits shift left. Code just decrements `n1`. No structural shift.
5. **Single sigma for decay at n_K.** Code discards at the last phase boundary. Notes say decay is continuous and probabilistic -- a qubit can decay at any phase, not just phase K.

### Open Questions
- Are the Erlang phases for decay or for service? Notes mention both `mu` (service/fidelity) and `sigma` (phase decay). Need clarification on how K-Erlang maps to the two processes.
- How does `sigma_i` relate to the fidelity function `F(t) = 1/2*(1+exp(-ct))`?
- When the notes say "hyperexponential", is that for comparison or for the baseline?
- What does "um bell pair herda o decaimento do qbit original" mean operationally in the CTMC?

## Session: 2026-05-01 -- Implementation (goodput_analysis.py)

### Model Summary

**System:** Single FIFO queue (Q=5 slots) representing qubits awaiting service.
Qubits arrive at rate `lambda`, are served at rate `mu`, and decay
probabilistically according to one of three distributions -- all with the
same mean lifetime `t* = -ln(2F-1)/c`.

**Queue semantics:** Compact (left-aligned). When any slot `i` is removed
(service at head, or decay anywhere), all slots to the right shift left,
and new empty slots (0) are appended at the tail. Slot 0 is always the head.

**Events per model:**

| Event | Description | Rate |
|---|---|---|
| Arrival | Finds first empty slot, inserts qubit | `lambda` |
| Service | Removes head (slot 0), shifts left | `mu` |
| Decay | Removes qubit at slot i, shifts left | distribution-dependent |

**Decay distributions (mean = t* in all three):**

| Type | Slot values | CV | Mechanics |
|---|---|---|---|
| Exponential | {0, 1} | 1.00 | One event per slot at rate `1/t*` |
| Erlang-2 | {0, 1, 2} | 0.71 | Two-phase chain: 1->2 at `2/t*`, 2->dead at `2/t*` |
| Hyperexponential | {0, 1-fast, 2-slow} | 1.62 | Arrival branches 50/50; fast decays at `10/t*`, slow at `10/(19*t*)` |

**Reachable states** (compact queue, only first N slots non-zero):

| Distribution | Values/slot | Formula | States |
|---|---|---|---|
| Exponential | 2 (0,1) | `sum(2^n, n=0..5)` | 6 |
| Erlang-2 | 2 decay states (1,2) | `sum(2^n, n=0..5)` | 63 |
| Hyper | 2 decay states (1,2) | `sum(2^n, n=0..5)` | 63 |

**Why exponential has only 6 states:** With values {0,1} and left-alignment,
the queue length uniquely identifies the state. No two patterns of the
same length differ (all non-zero slots are `1`).

**Why Erlang-2 and Hyper have 63 (not 243):** Of the 3 possible values
per slot {0,1,2}, only two are "occupied" states (1 and 2). The 3^5 = 243
upper bound is unattainable because left-alignment forbids gaps like
`[1,0,1,0,0]`. Reachable count = `sum(2^n, n=0..Q)` = 63.

**Solving:** `model.solve()` -- BFS state-space exploration, Q-matrix assembly,
analytical steady-state (`piQ = 0`). No simulation. Goodput = `pi . reward_vector`
(impulse reward on the service event).

**Metrics:**
- Goodput = fraction of arrivals that complete service before decaying
- Utility = Fidelity threshold x Goodput rate
- CV = Coefficient of Variation of the decay lifetime

**Plot:** `goodput_vs_fidelity.png` — two stacked subplots (Goodput, Utility) with constants in title.

### Model Built
Single file: `goodput_analysis.py`. Three analytically solvable models (all pure EXP events):
- **Exponential**: slot values {0, 1}, decay at rate `1/t*`
- **Erlang-2**: slot values {0, 1, 2}, phase advance at `2/t*`, decay at `2/t*`
- **Hyperexponential (p=0.5)**: slot values {0, 1-fast, 2-slow}, arrival branches 50/50, decay at `10/t*` and `10/(19*t*)`

### Parameters
- Q=10 (slots), lambda=1.0, mu=2.0, c=0.1
- Fidelity sweep: F from 0.505 to 0.999 (100 points)
- t* = -ln(2F-1)/c
- Goodput% = (goodput_rate / lambda) * 100
- Utility = F * goodput_rate

### State Spaces (compact queue, Q=10)
- Exponential: 11 states
- Erlang-2: 2047 states
- Hyperexponential: 2047 states

### Utility maxima (F x goodput_rate)
| Distribution | CV  | Peak F | Peak Utility |
|---|---|---|---|
| Erlang-2 | 0.71 | 0.859 | 0.766 |
| Exponential | 1.00 | 0.844 | 0.698 |
| Hyperexponential | 1.62 | 0.789 | 0.541 |

### Confirmation
Lower CV → higher peak utility, higher optimal fidelity. Erlang-2 sustains
higher fidelity thresholds before decay kills throughput. Hyperexponential
(highest variance) peaks earliest and lowest.
