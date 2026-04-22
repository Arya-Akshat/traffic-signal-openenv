# Round 2 Plan: Hierarchical Multi-Agent Traffic Control OpenEnv

## 0. Purpose of this document

This document is the long-term build plan for the Round 2 version of the project. It is meant to preserve project context even if the chat is lost. It should be treated as the canonical roadmap for implementation, evaluation, and pitch preparation.

The project already exists as a working OpenEnv-style traffic signal environment. Round 2 will evolve that baseline into a more ambitious **hierarchical multi-agent traffic control system**.

This plan describes:

* what the project is trying to model in the real world,
* how the central controller and local controllers should work,
* what the environment state and actions should look like,
* how rewards and graders should be designed,
* what we should build first, next, and later,
* and how we will prove improvement during the hackathon.

---

## 1. Current project baseline

The existing project is a deterministic traffic signal environment exposed through an OpenEnv-compatible HTTP API.

Current baseline capabilities:

* `POST /reset`
* `POST /step`
* `GET /state`
* `GET /health`
* `GET /metadata`
* `GET /schema`
* `POST /mcp`
* Docker deployment on Hugging Face Spaces
* task profiles for easy, medium, and hard traffic conditions
* deterministic transitions with queue length and waiting time dynamics
* per-step reward and task-level score/grade logic
* baseline `inference.py` for automated runs

The Round 2 version should keep this core and expand it into a richer system with **hierarchy**, **coordination**, and **heterogeneous local behavior**.

---

## 2. Round 2 project direction

### Main idea

We will transform the project from:

> one controller managing one traffic simulation

into:

> a hierarchical multi-agent traffic system where a central controller coordinates multiple local intersection controllers.

### Real-world intuition

This is inspired by real traffic management systems:

* local controllers make fast decisions at each intersection,
* a central policy layer monitors city-wide congestion,
* the central layer adjusts priorities, policies, and global constraints,
* local layers still act independently in short time scales,
* the system must work under uncertainty, emergency vehicles, lane imbalances, and partial observability.

This is much closer to a real urban traffic control problem than a single flat agent.

---

## 3. Why this is the right direction

This project fits the Round 2 themes well:

### Theme alignment

* **Theme 1: Multi-Agent Interactions**

  * multiple local controllers
  * coordination between intersections
  * global policy shaping local behavior

* **Theme 2: Long-Horizon Planning**

  * local decisions affect traffic many steps later
  * congestion can accumulate across time and across intersections
  * a central controller must think beyond immediate reward

* **Theme 3: World Modeling**

  * state changes over time
  * partial observability is realistic
  * local agents do not see everything
  * the central agent must reason over system-wide conditions

### Why this is better than restarting from scratch

The current project already has:

* a working OpenEnv base,
* deployment,
* deterministic logic,
* tasks and graders,
* an inference script,
* and a full submission pipeline.

Starting from scratch would waste this foundation. The best move is to **upgrade the existing project** into a more complex and more theme-aligned system.

---

## 4. Final concept for Round 2

### Short description

A city traffic environment where:

* each intersection is controlled by a local controller,
* a central controller observes macro-level traffic and adjusts system policy,
* local controllers optimize their own junctions,
* the central controller optimizes the overall network,
* both levels must cooperate under uncertainty and emergency events.

### High-level story

The agent is not just “turn the light green.”
The agent is now part of a **traffic orchestration system**:

* local agents handle their own intersection timing,
* a central agent manages global congestion, policy weights, and special events,
* the overall objective is to reduce waiting time, prevent bottlenecks, and maintain corridor flow.

---

## 5. Hierarchical multi-agent design

## 5.1 Roles

### Central controller

The central controller is the global policy layer. Its job is not to directly micromanage every lane at every time step, but to influence the local agents through policy parameters and strategic decisions.

Its responsibilities should include:

* monitoring aggregate congestion across all intersections,
* detecting imbalanced traffic corridors,
* adjusting local policy priorities,
* assigning weights to intersections based on global load,
* increasing emergency priority when needed,
* managing corridor-level coordination,
* deciding when certain intersections should become more aggressive or more conservative,
* preventing local optimization from harming the network.

### Local controllers

Each local controller manages one intersection.

Its responsibilities should include:

* selecting signal phases,
* handling short-term queue pressure,
* reducing local waiting time,
* responding to the central policy adjustments,
* exploiting local information that the central controller does not see in detail.

Local controllers are the “fast loop.”
The central controller is the “slow loop.”

---

## 6. Real-world logic for the central and local layers

## 6.1 Local controller logic

Each intersection should behave like a real local traffic signal controller.

It should observe:

* queue lengths for the lanes at that intersection,
* waiting times,
* current phase,
* time spent in the current phase,
* maybe a small amount of recent history.

It should decide:

* whether to keep the current phase,
* whether to switch,
* which lane group should receive green next,
* whether to prioritize clearing the currently longest queue,
* whether to hold green a little longer if a queue is still large.

This should be done using a local objective such as:

* minimizing queue buildup,
* reducing waiting time,
* preventing unnecessary switching,
* responding quickly to spikes.

### Local controller rewards

Local rewards should focus on:

* queue reduction,
* waiting time reduction,
* avoiding phase thrashing,
* short-term throughput.

This makes the local controller good at immediate tactical control.

---

## 6.2 Central controller logic

The central controller should not choose every local light directly. Instead, it should decide **policy settings** that modify how local controllers behave.

Examples of central policy decisions:

* increase/decrease switch penalties for certain intersections,
* assign priority to a corridor when downstream congestion is high,
* add emergency vehicle priority to a specific region,
* temporarily make one intersection more aggressive while another becomes conservative,
* redistribute weights between local and global objectives,
* set target balance ratios across adjacent intersections,
* signal a “rush hour” policy mode or “incident response” policy mode.

The central controller should optimize:

* global throughput,
* corridor balance,
* emergency handling,
* avoiding spillback,
* fairness between intersections,
* network-wide efficiency.

### Central controller reward

The central reward should be based on:

* total network waiting time,
* total queue across all intersections,
* average throughput,
* imbalance penalties,
* emergency response success,
* corridor smoothness.

This gives the central controller a more strategic objective than the local agents.

---

## 7. How the two levels interact

The key idea is that the central controller changes the conditions under which the local controllers operate.

### Example interaction cycle

1. Local controllers observe their own intersection states.
2. They select immediate traffic phase actions.
3. The environment updates all intersections.
4. The central controller observes system-wide metrics.
5. The central controller updates a policy vector or set of weights.
6. Those policy changes affect the next local decisions.

### Example policy parameters that the central controller may adjust

* switching penalties,
* lane priority multipliers,
* emergency boost factors,
* queue urgency weights,
* corridor coordination weights,
* phase duration targets,
* balancing penalties for cross-intersection asymmetry.

This creates true hierarchy:

* local controllers act fast,
* the central controller acts strategically,
* the whole system becomes multi-agent and multi-timescale.

---

## 8. Proposed environment structure

## 8.1 City-level topology

The environment should evolve from one intersection into a small network.

Recommended starting topology:

* 2x2 grid of intersections
* later expandable to a corridor or 3x3 grid
* each intersection has 4 incoming lanes
* traffic can propagate from one intersection to another

### Why this matters

A single intersection is too easy.
A grid or corridor creates:

* upstream/downstream coupling,
* spillback,
* bottlenecks,
* global coordination challenges,
* more interesting behaviors.

---

## 8.2 Agent mapping

Recommended hierarchy:

* **1 central controller**

  * sees aggregated city state
  * updates global policy parameters

* **N local controllers**

  * one per intersection
  * receives local state + central policy context
  * chooses local phase actions

This can still be represented in one OpenEnv environment for hackathon purposes, but conceptually it should be multi-agent.

---

## 9. Inputs and outputs

## 9.1 Input to the environment

The environment should accept:

* `task_id` at reset time,
* action(s) from local controllers,
* central policy action or policy update,
* optional scenario modifiers such as emergency events.

### Example action types

Local actions:

* keep current phase
* switch to next phase
* choose explicit phase

Central actions:

* increase corridor priority
* increase emergency priority
* reduce switching in congested region
* rebalance policy weights
* push green wave timing forward

---

## 9.2 Output from the environment

The environment should return:

* local observations,
* global observations,
* step reward,
* done flag,
* info dictionary,
* episode-level metrics,
* task score.

### Important metrics to expose

* queue lengths per lane/intersection,
* waiting time per lane/intersection,
* current phase per intersection,
* phase timer,
* total waiting time,
* total throughput,
* corridor imbalance score,
* emergency delay score,
* central policy status.

---

## 10. Task design

We should preserve the existing easy / medium / hard concept, but make the differences more meaningful and network-aware.

## 10.1 Easy task

Goal:

* single corridor or simple 2-intersection setup
* predictable arrivals
* low randomness
* no emergencies

What it teaches:

* basic local phase selection
* queue reduction
* avoiding unnecessary switches

## 10.2 Medium task

Goal:

* multiple intersections
* random traffic spikes
* uneven demand across directions
* mild coordination needs

What it teaches:

* local + central coordination
* handling fluctuations
* balancing fairness and throughput

## 10.3 Hard task

Goal:

* larger network or 2x2 grid
* strong asymmetry in demand
* emergency vehicles
* spillback risk
* delayed consequences

What it teaches:

* long-horizon planning
* coordination under uncertainty
* emergency-aware optimization
* robust network-level control

---

## 11. Reward design

The reward system should be layered.

## 11.1 Local reward

For each local intersection:

* reward should prefer lower queues,
* lower waiting times,
* higher lane throughput,
* stable phases,
* fewer pointless switches.

## 11.2 Central reward

For the central policy:

* reward should prefer globally lower congestion,
* better corridor balance,
* emergency response speed,
* fewer spillbacks,
* smooth system-wide flow.

## 11.3 Combined reward idea

A practical version could be:

`total_reward = local_reward + alpha * central_reward - penalty_for_instability`

Where:

* `alpha` controls the influence of global coordination,
* instability penalty prevents overreaction,
* special emergency bonuses reward quick response.

---

## 12. Grading design

We already have grading infrastructure. For Round 2, the graders should measure more than just queue reduction.

### Grader inputs

* average waiting time,
* throughput,
* queue length imbalance,
* emergency delay,
* global flow smoothness,
* policy stability,
* maybe coordination efficiency across intersections.

### Grader output

* final score in `(0, 1)`,
* never exactly 0 or 1,
* reproducible and deterministic.

### Suggested grading philosophy

* easy task: score based on local efficiency,
* medium task: score based on mixed local + coordination performance,
* hard task: score based on handling uncertainty and preserving network stability.

---

## 13. Inference and training plan

## 13.1 Baseline inference

The inference script should:

* run the environment episode(s),
* use an OpenAI-compatible LLM client if enabled,
* fall back to a deterministic rule-based policy,
* print the required structured logs,
* remain stable under Docker and HF deployment.

## 13.2 Training path

The minimum training demonstration should show:

* baseline controller performance,
* improved controller performance after light training or tuning,
* visible reward improvement curves.

### Recommended training story

* before: purely local heuristic controller,
* after: central policy adjusts local behavior and improves global score.

This is important for the hackathon pitch.

---

## 14. What makes this project competitive

To be highly competitive, the environment must show:

### 1. Realism

* city traffic is a real-world scheduling problem,
* local and global control is realistic,
* emergency handling is realistic.

### 2. Complexity

* multiple intersections,
* delayed effects,
* policy coupling,
* heterogeneous tasks.

### 3. Measurable improvement

* baseline vs improved controller,
* reward curves,
* reduced waiting time,
* better throughput.

### 4. Storytelling value

* easy to explain,
* visually obvious in a demo,
* intuitive to judges.

---

## 15. Risks to avoid

### Risk 1: fake multi-agent design

If the central controller does not actually affect local controllers, the multi-agent framing becomes weak.

### Risk 2: too much complexity too early

Do not implement a giant city before proving the 2x2 or corridor version works.

### Risk 3: reward hacking

If the policy learns to exploit the scoring function instead of solving traffic, the benchmark becomes weak.

### Risk 4: not showing improvement

The demo must clearly show that training or policy changes improve metrics.

### Risk 5: losing determinism

Deterministic reproducibility is critical for debugging and judging.

---

## 16. Development phases

## Phase A — Stabilize current base

* keep current OpenEnv endpoints stable,
* preserve Docker deployment,
* preserve current task structure,
* ensure current tests still pass.

## Phase B — Introduce hierarchical abstraction

* define central controller layer,
* define local controller layer,
* map current single-intersection logic into a multi-controller model.

## Phase C — Expand environment topology

* create a small multi-intersection grid or corridor,
* add downstream coupling.

## Phase D — Add coordination pressure

* emergency vehicles,
* spillback,
* route imbalance,
* local/global trade-offs.

## Phase E — Improve scoring and evaluation

* refine graders,
* add metrics,
* show improvement curves.

## Phase F — Demo and pitch

* produce a clean story,
* show before/after,
* explain why the environment matters.

---

## 17. Suggested implementation order

### First

* finalize the concept and naming,
* keep the current OpenEnv scaffold,
* define the central/local hierarchy in the model.

### Next

* add a multi-intersection state model,
* add policy override parameters,
* modify the reward to include global coordination.

### Then

* add scenario-specific tasks,
* make hard task meaningfully different.

### Finally

* tune graders,
* run training/inference baseline,
* prepare the pitch material.

---

## 18. Recommended project narrative

Use a story like this:

> We started with a traffic signal OpenEnv environment, and evolved it into a hierarchical multi-agent traffic orchestration benchmark. Local agents control individual intersections, while a central controller adjusts global policy to handle corridor congestion, spikes, and emergencies. This models real-world traffic management better than a single monolithic agent.

This is a strong pitch because it is:

* understandable,
* realistic,
* complex,
* and clearly connected to the theme.

---

## 19. Future extensions

Possible future additions if time allows:

* traffic incidents and blocked lanes,
* bus priority lanes,
* ambulance dispatch route control,
* communication delays between controllers,
* learning-based central policy updates,
* adaptive curricula for self-improvement,
* tool-based traffic simulation integration,
* partial observability masks per controller.

---

## 20. Final goal for Round 2

The final goal is not just to have a working environment.
The final goal is to have a **strong, realistic, theme-aligned benchmark** that can be used to train and compare agents on hierarchical city-scale traffic control.

Success means:

* the environment is stable,
* the hierarchy is real,
* the tasks are different and meaningful,
* the scores show improvement,
* and the judges can immediately understand why the project matters.

---

## 21. Keep this document updated

This file should be updated whenever:

* the environment structure changes,
* new tasks are added,
* the reward function changes,
* the grader logic changes,
* new deployment requirements appear,
* or the pitch narrative changes.

If you lose the chat, this document should be enough to resume the project.
