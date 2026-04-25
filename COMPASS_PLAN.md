# COMPASS: Conformation-Map Optimized Mutant Prioritized Adaptive Sequential Sampling

**Project:** BTP
**By:** Anmol Goel 
**Version:** 1
**Source / prior system:** Long MD of a related peptide or protein variant, currently A10  
**Target / mutant system:** One starting target structure, currently A8Q2 for validation  
**Method name:** COMPASS — **C**onformation-map **O**ptimized **M**utant **P**rioritized **A**daptive **S**equential **S**ampling  

**Central scientific question:**  
Can a long MD trajectory of one sequence be used as a structured prior to accelerate sampling of a related mutant, when the mutant starts from only one available structure and every short MD segment is expensive?

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation](#2-motivation)
3. [Analogies and Their Algorithmic Meaning](#3-analogies-and-their-algorithmic-meaning)
4. [First-Principles View](#4-first-principles-view)
5. [Core Assumptions](#5-core-assumptions)
6. [What the Source MD Gives Us](#6-what-the-source-md-gives-us)
7. [What the Target Starts With](#7-what-the-target-starts-with)
8. [The Main Object: A Mutable Belief State](#8-the-main-object-a-mutable-belief-state)
9. [Probe Types](#9-probe-types)
10. [Sequential Decision Score](#10-sequential-decision-score)
11. [Exploration vs Exploitation](#11-exploration-vs-exploitation)
12. [Full Algorithm](#12-full-algorithm)
13. [Validation on A10 -> A8Q2](#13-validation-on-a10---a8q2)
14. [Findings From the COMPASS-Lite Replay Sweep](#14-findings-from-the-compass-lite-replay-sweep)
15. [BTP Presentation Scope](#15-btp-presentation-scope)
16. [Metrics and Scientific Claims](#16-metrics-and-scientific-claims)
17. [Implementation Plan in This Repository](#17-implementation-plan-in-this-repository)
18. [Failure Modes and Caveats](#18-failure-modes-and-caveats)
19. [Open Design Questions](#19-open-design-questions)
20. [References and Related Ideas](#20-references-and-related-ideas)

---

## 1. Executive Summary

The existing experiment asked whether an A10-trained TICA coordinate transfers to A8Q2. The outcome was nuanced: formal transfer metrics mostly failed, but the transferred coordinate still retained a strong helix-content signal and helped estimate helix fraction. This suggests that transfer is useful, but only if it is treated as a fallible prior rather than a fixed truth.

This document proposes a stricter and more compute-aware method:

> Every target MD segment should answer a specific question about how the mutant landscape differs from the source landscape.

The method is sequential, not parallel-first. At each decision point, the algorithm chooses exactly one next target frame to seed from. The chosen frame is the one expected to reduce the most important uncertainty per unit of compute.

The source MD is used to build an atlas: states, transition routes, slow coordinates, basin depths, structural features, and likely bottlenecks. The target simulation gradually updates trust in each part of this atlas. Some source basins may be confirmed cheaply. Some old transition routes may remain open. Some routes may be blocked by the mutation. Some target-only frontiers may appear and must be explored without relying on the source.

The novelty is not simply "use transferred TICA." The novelty is:

> Use a long source trajectory as a structured prior, then adaptively decide whether to confirm, exploit, reject, or ignore each part of that prior using the cheapest possible target probes.

---

## 2. Motivation

### 2.1 Compute Is the Central Constraint

For expensive MD, even 100 ps should not be spent casually. A single 1000 ps target run is often less logical than ten 100 ps target probes, if each probe is chosen to test a different hypothesis.

Randomness already exists through thermal velocities and stochastic thermostats. The algorithm should not add randomness through careless seed choice. The target landscape is unknown, but the source landscape is not. The central task is to spend target compute where the source prior is most useful or most questionable.

### 2.2 Practical Real-World Setting

The real setting is not always A10 -> A8Q2 with both ground truths available. A more realistic setting is:

1. A long MD trajectory exists for one related sequence, mutant, homolog, ligand state, protonation state, or force-field condition.
2. A new target system has only one initial structure, possibly from AlphaFold, homology modeling, or an experimental structure.
3. The target is believed to have a similar global landscape but not identical basin depths, barriers, or transition routes.
4. The goal is to learn the target landscape far faster than running a fresh long target MD trajectory.

A10 -> A8Q2 is therefore a validation sandbox. A8Q2 ground truth already exists on disk, so it can be used to measure whether this method actually accelerates convergence.


## 3. Analogies and Their Algorithmic Meaning

Analogies are useful here because they expose different aspects of the sampling problem.

### 3.1 The Foggy City Map

We have an old map of a foggy city. A hiker moves through the city slowly. A drone can observe and airdrop the hiker, but only to locations that are considered safe or already observed. The old map was created before a large city change. Many roads and hills remain similar, but some tunnels are blocked, some valleys shifted, and some routes became harder or easier.

Mapping:

| Analogy | MD object |
|---|---|
| City terrain | Free energy landscape |
| Hills | Barriers |
| Valleys | Metastable states |
| Roads/tunnels/passes | Transition pathways |
| Hiker | Unbiased MD trajectory |
| Hiker laziness | Rare thermal barrier crossing |
| Drone airdrop | Restarting MD from selected frames |
| Fog | Unknown target landscape |
| Old map | Source MD atlas |
| City change | Mutation or perturbation |
| Blocked tunnel | Source transition route that fails in target |
| Newly opened route | Target-only transition not predicted by source |

Algorithmic lesson:

- The old map should guide us, but not control us.
- The best next move is not always the most unexplored place.
- Sometimes the best move is to cheaply confirm that an important old district is unchanged.
- Sometimes the best move is to test whether an old tunnel still works.
- Sometimes the best move is to ignore the old map and follow a new target frontier.

### 3.2 The Modified Spring-Magnet Machine

Imagine a small mechanical device made of rods, springs, hinges, and magnets. It is shaken in a box and flips between stable shapes. We watched the original machine for hours and learned its full behavior. Now one arm length changes or one magnet moves. The device remains similar, but its stable shapes and switching routes may shift.

Mapping:

| Analogy | MD object |
|---|---|
| Machine components | Residues, contacts, torsions |
| Springs and rods | Bonded and nonbonded constraints |
| Magnets | Stabilizing interactions, H-bonds, side-chain contacts |
| Shaking box | Thermal bath |
| Stable machine shapes | Conformational basins |
| Modified component | Mutation |
| Watching old machine | Source MD trajectory |
| Poking in a direction | Short target MD probe |

Algorithmic lesson:

- The perturbation is local in component space but can have global effects.
- If a modified component was irrelevant to a basin, that basin may remain valid.
- If a modified component was load-bearing for a transition route, that route may change drastically.
- The source trajectory can tell us which regions depended on the mutated component.

This analogy suggests a useful prior: source regions strongly involving mutated residues should start with lower trust than regions controlled by conserved features.

### 3.3 Combined Lesson

The map analogy emphasizes topology: basins, passes, routes, and frontiers.

The machine analogy emphasizes mechanism: which local interactions support which global states.

Together they imply:

> The target landscape is probably not random. It is a structured deformation of the source landscape. The algorithm should learn the deformation with as few target probes as possible.

---

## 4. First-Principles View

### 4.1 MD Sampling as Sequential Experiment Design

At any time, the algorithm has:

- a source prior,
- a small target dataset,
- uncertainty about target basin populations,
- uncertainty about target transitions,
- uncertainty about which source assumptions remain valid.

The next MD segment should be chosen as an experiment. The experiment has a cost, usually simulation length. It has an expected information value.

The decision problem is:

```text
Choose next seed x and short duration tau
to maximize expected useful information gained per unit compute.
```

Useful information depends on the scientific goal. If the goal is FES reconstruction, useful information means reducing uncertainty in important basin populations and coverage. If the goal is first transition discovery, useful information means finding open routes. If the goal is a thermodynamic observable such as helix fraction, useful information means reducing uncertainty near the boundary between helix and coil.

### 4.2 Why Short Probes Can Beat One Longer Run

A single long run is locally continuous. If it starts in a deep basin, it may spend most compute repeatedly confirming that the basin is stable. Multiple short targeted probes can test multiple hypotheses:

- Does the starting structure relax into the expected source basin?
- Is the nearest source exit still open?
- Is the target drifting toward a different basin than the source predicts?
- Is a suspected high-value basin reachable from the current frontier?
- Is the source prior failing in this region?

Each question can be answered partially by a short segment. No single short segment is enough to solve the target landscape, but the sequence of probes can update the belief state quickly.

### 4.3 What "Acceleration" Means

There are multiple speedups:

1. **Wall-clock speedup:** reduced time due to parallelism or shorter depth per trajectory.
2. **Aggregate-compute speedup:** fewer total ns to reach a target error.
3. **Discovery speedup:** shorter ns until first discovery of a basin or transition.
4. **Thermodynamic speedup:** shorter ns until an observable like f_helix converges.
5. **FES speedup:** shorter ns until JSD from ground truth reaches a threshold.

This proposal focuses on aggregate-compute speedup and discovery speedup. Parallelism is useful but should not be the main claim. Even if everything is run sequentially, the method should choose better next experiments than random or long unbiased MD.

---

## 5. Core Assumptions

### 5.1 Landscape Similarity

The target landscape is assumed to be substantially similar to the source landscape. This does not mean identical. It means:

- many metastable states have source analogs,
- many slow modes remain useful,
- transition routes are often related,
- basin depths and barriers can shift,
- some routes may be blocked or newly opened.

For A10 -> A8Q2 this is plausible because the mutation is small relative to the peptide length and the main helix-coil process remains shared.

### 5.2 Target Seeds Are Only Target-Visited Frames in Version 1

Version 1 should not directly mutate source structures and equilibrate them as target seeds. This keeps the experiment conservative:

- every restart is from a physically visited target frame,
- no source reservoir insertion is needed,
- analysis remains closer to unbiased restart adaptive sampling.

This restriction limits maximum speedup. A future version may add source-structure reservoir insertion with strict validation and reweighting.

### 5.3 The Source Prior Is Fallible

The source prior is not assumed correct. It is a starting belief. It can be locally confirmed, downgraded, or rejected.

Examples:

- A source basin may remain geometrically valid but change population.
- A source coordinate may remain correlated with helix content but fail as a faithful kinetic coordinate.
- A source transition route may become much harder in the target.
- A source low-population region may become important in the target.

### 5.4 Mutant-Only Exploration Is Required

Because transferred information can fail, the algorithm must include an explicit source-independent exploration term. Existing A8Q2 results show that random/adaptive sampling is a strong baseline. Any source-guided method must beat or complement that, not replace it blindly.

### 5.5 Short Probes Need Local Questions

A 50-100 ps probe cannot estimate global kinetics. It can estimate:

- local relaxation direction,
- local spread,
- whether a selected frame immediately falls back to a basin,
- whether it crosses a nearby boundary,
- whether it moves along or against a source-predicted route.

The scoring system should ask questions that short probes can actually answer.

---

## 6. What the Source MD Gives Us

The source trajectory provides multiple layers of information. Each layer can be used separately and trusted separately.

### 6.1 Coordinate System

The source trajectory provides a low-dimensional map:

- TICA coordinates,
- SPIB latent coordinates,
- RMSD / NHB / helix-content axes,
- selected contact or torsion features.

The target frames can be projected into this coordinate system if equivalent features can be computed. We will use unsupervised coordinates only since in real world we might not have features like rmsd, hhb, helix content or selected contact and torsion features. 

### 6.2 State Decomposition

The source trajectory can be clustered into states:

- metastable basins,
- microstates,
- FES bins,
- helix/coil/intermediate labels.

These state labels become the initial map. The target does not need enough data to build a native state decomposition from scratch. We don't use anything that is supervised in nature and require extra human inputs. 

### 6.3 Free Energy Prior

The source provides approximate basin depths:

```text
F_source(z) = -kT log p_source(z)
```

This tells us:

- where source basins are,
- which basins are rare,
- which source regions are transition-like,
- what range of the map is physically relevant.

### 6.4 Transition Prior

The source transition matrix or observed transition graph tells us:

- which states connect,
- which exits are common,
- which exits are rare,
- which bottlenecks matter for slow dynamics.

This is the "roads and tunnels" layer of the map.

### 6.5 Structural Reservoir, Not Used in Version 1

The source also contains physical structures from all old regions. Version 1 does not directly use these as seeds. They can still be used for analysis:

- define source basin centers,
- define target direction-to-basin vectors,
- estimate source structural variability,
- measure whether target frames are in-domain or out-of-domain.

### 6.6 Mutation Sensitivity Prior

The source trajectory can estimate which regions depend on residues changed in the target. For A10 -> A8Q2, residues near positions 9-10 or C-terminal helix interactions may be especially relevant.

Possible mutation-sensitivity features:

- mutual information between residue-local features and source state labels,
- contact frequencies involving mutated positions,
- H-bond involvement by mutated positions,
- per-state side-chain exposure or local packing,
- per-state contribution of terminal residues to helix order.

Regions with high mutation sensitivity start with lower source trust.

---

## 7. What the Target Starts With

In the real-world setting, the target starts with:

- one target structure,
- a topology and force field,
- equivalent feature definitions,
- source atlas,
- no target ground truth.

In the A10 -> A8Q2 validation setting, target ground truth exists but should be hidden from the adaptive algorithm. It is only used afterward for scoring.

The first target probe should not be random. It answers:

1. Does the target starting structure lie inside the source atlas?
2. Which source state is it closest to?
3. Does it relax toward the expected source basin or drift away?
4. Does it remain locally stable under target dynamics?
5. Which target frames are now legal restart candidates?

Even the first 50-100 ps segment has a job.

---

## 8. The Main Object: A Mutable Belief State

The algorithm maintains a belief state over source-defined regions and routes. This belief state is updated after every target probe.

### 8.1 Region-Level Quantities

For each source region or bin `r`:

```text
visited_target(r)           # number of target frames in region
density_target(r)           # target occupancy estimate
density_source(r)           # source occupancy estimate
trust_region(r)             # how much source behavior is trusted here
disagreement_region(r)      # source-target mismatch
mutation_sensitivity(r)     # prior risk that mutation affects this region
importance(r)               # relevance to target observable / FES / connectivity
frontier_target(r)          # whether target samples lie on edge of explored cloud
local_uncertainty(r)        # posterior uncertainty about target behavior
```

### 8.2 Route-Level Quantities

For each source route or transition `i -> j`:

```text
source_transition_prob(i,j)
target_success_count(i,j)
target_failure_count(i,j)
escape_posterior(i,j)
route_trust(i,j)
route_importance(i,j)
```

The escape posterior can be modeled as a Beta distribution:

```text
p_escape_target(i,j) ~ Beta(a_ij, b_ij)
```

The source transition probability initializes the prior. Successful target crossings increase `a_ij`; failed attempts increase `b_ij`.

This is a principled version of decay-penalized replay:

- repeated failed attempts reduce the route score,
- one successful attempt can reopen confidence,
- the algorithm does not waste compute forever on blocked exits.

### 8.3 Candidate Frame Quantities

For every target frame `x` already generated:

```text
z(x)                    # source coordinate projection
region(x)               # source region/bin label
nearest_source_state(x)
distance_to_source_support(x)
local_target_density(x)
frontier_score(x)
disagreement_score(x)
redundancy_score(x)
```

Candidate seeds are always target-visited frames in Version 1.

---

## 9. Probe Types

The algorithm chooses among five conceptual probe types. They are not separate algorithms; they are terms in one decision score.

### 9.1 Confirm a Cheap Basin

**Question:** Is this source basin still basically valid in the target?

Use when:

- source says the basin is important,
- target has not tested it enough,
- source trust is uncertain but not already contradicted,
- mutation sensitivity is low or moderate,
- a short probe can answer local stability.

Why this can be the best next step:

Confirming an important basin cheaply may eliminate a large uncertainty. If a source basin is unchanged, the algorithm can stop spending much target compute there and borrow source understanding cautiously.

This is exploitation of the source prior, but it is conservative exploitation.

### 9.2 Probe a Suspect Saddle

**Question:** Is this old source pass still open in the target?

Use when:

- source route connects important basins,
- target is currently near the source route,
- mutation sensitivity suggests route may have changed,
- route escape probability is uncertain,
- failed attempts have not already made the route implausible.

Why this can be the best next step:

If the route is open, one short probe can discover a valuable transition. If the route is closed, learning that also matters because it downgrades the source graph.

This is testing the source "road map."

### 9.3 Probe a Promising Tunnel

**Question:** Does the target's own drift suggest a path toward a new or under-sampled state?

Use when:

- recent target probes show consistent movement in a direction,
- the target cloud is elongated toward an unvisited source region,
- the current frame is on the frontier of target samples,
- source topology says the direction is plausible.

Why this can be the best next step:

This is neither blind source exploitation nor blind target exploration. It follows the target's local evidence while using source topology to interpret where the path may lead.

This may be the most important mode for compute efficiency.

### 9.4 Probe an Unvisited Source Basin Direction

**Question:** Can the target reach an important source basin that has not yet been observed?

Use when:

- source has an important basin not yet visited by target,
- there is a target frame closest to the route toward that basin,
- the path has not repeatedly failed,
- current sampled basins are already over-confirmed.

Why this can be the best next step:

It avoids spending all compute near the initial structure. It uses source knowledge of what regions probably exist, without directly inserting source structures.

### 9.5 Probe Target Frontier or Highest Disagreement

**Question:** What is the target doing that the source does not explain?

Use when:

- target frames are out-of-distribution relative to source,
- local source-target disagreement is high,
- source trust is low,
- target frontier is expanding in a direction not predicted by source,
- random/adaptive exploration should be allowed to compete.

Why this can be the best next step:

This prevents transfer bias. If the mutation changed the landscape in a real way, the algorithm must discover that rather than force the target into source-defined routes.

---

## 10. Sequential Decision Score

At each iteration, score every eligible target frame `x`.

```text
TotalScore(x) =
    w_confirm   * ConfirmScore(x)
  + w_saddle    * SaddleScore(x)
  + w_tunnel    * TunnelScore(x)
  + w_unvisited * UnvisitedBasinScore(x)
  + w_frontier  * FrontierDisagreementScore(x)
  - w_repeat    * RedundancyPenalty(x)
  - w_fail      * LocalFailurePenalty(x)
```

The highest-scoring frame becomes the next seed.

### 10.1 Confirm Score

```text
ConfirmScore(x) =
    importance(region(x))
  * uncertainty_trust(region(x))
  * expected_source_validity(region(x))
  * cheapness(region(x))
```

Interpretation:

- Important basins deserve confirmation.
- Already-confirmed basins should not be repeatedly sampled.
- Very suspect regions are better handled by disagreement/frontier terms than by cheap confirmation.

### 10.2 Saddle Score

For candidate frame `x` near a source route `i -> j`:

```text
SaddleScore(x) =
    source_saddle_score(x)
  * route_importance(i,j)
  * trust_region(i)
  * uncertainty_escape(i,j)
  * mean_escape_posterior(i,j)
```

With failure decay:

```text
mean_escape_posterior(i,j) = a_ij / (a_ij + b_ij)
```

Repeated failed attempts lower this term automatically.

### 10.3 Tunnel Score

```text
TunnelScore(x) =
    frontier_target(x)
  * alignment(local_target_drift(x), direction_to_high_value_region(x))
  * target_drift_confidence(x)
  * route_plausibility_source(x)
```

This term asks whether target dynamics are already hinting at a productive direction.

### 10.4 Unvisited Basin Score

```text
UnvisitedBasinScore(x) =
    importance(target_source_basin)
  * not_visited_target(target_source_basin)
  * proximity_to_route(x, target_source_basin)
  * trust_or_uncertainty_of_route(x, target_source_basin)
  * (1 - route_failure_probability)
```

This term points toward important old basins that target has not reached yet.

### 10.5 Frontier Disagreement Score

```text
FrontierDisagreementScore(x) =
    frontier_target(x)
  * novelty_target(x)
  * disagreement_region(region(x))
  * local_model_uncertainty(region(x))
```

This term protects against source-prior blindness.

### 10.6 Penalties

```text
RedundancyPenalty(x) =
    similarity_to_recent_seeds(x)
  + same_region_overuse(region(x))
```

```text
LocalFailurePenalty(x) =
    failures_from_region(region(x))
  + failures_on_nearest_route(x)
```

These prevent repeatedly launching from nearly identical frames or repeatedly attacking a route that has already failed.

### 10.7 Two-Axis Decomposition of the Score (added after the sweep)

After running the sweep in Section 14, the six probe-type terms collapse, from a first-principles point of view, into **two orthogonal axes**:

```text
DENSITY-MATCHING axis
  = ConfirmScore + UnvisitedBasinScore restricted to high-density source bins
  = "fill bins in proportion to the source prior's density"

FRONTIER-EXPANSION axis
  = SaddleScore + TunnelScore + FrontierDisagreementScore
  = "push into low-density / out-of-support / target-novel bins"
```

This decomposition explains everything we observed in the sweep:

1. The density-matching axis cannot beat random on JSD-vs-marginal in retrospective replay, because random sampling of an i.i.d. chunk pool is the Bayes-optimal estimator of that marginal. Density-matching can only match it.
2. The frontier-expansion axis maximises coverage at the cost of distorting the marginal (because it deliberately oversamples low-density bins).
3. Where the prior helps in retrospective replay is on **rare-state observables**. Density-matching reduces the variance of estimators that depend on a small set of bins (helix fraction, contact frequencies), because it visits those bins earlier than i.i.d. would.
4. Where the prior helps in **online MD** (not tested in the replay) is in the early-budget exploration of regions a single-trajectory baseline would not reach. That benefit is invisible in replay and must be argued separately.

The asymptotic recipe that follows from this decomposition is **explore-then-exploit on the budget**, not on iteration count. The explicit weight schedule is in Section 11.

---

## 11. Exploration vs Exploitation

Exploration and exploitation are not fixed phases. They are consequences of the belief state.

### 11.1 Exploit the Source Prior When

Exploit source information when:

- source trust is high,
- the region or route is important,
- target evidence does not contradict source behavior,
- a route is likely open,
- the expected reward is high.

Typical action:

- probe a source saddle,
- follow a source-predicted route,
- move toward an unvisited source basin.

### 11.2 Explore the Target When

Explore target-only behavior when:

- source-target disagreement is high,
- target frames move out of source support,
- repeated attempts on source routes fail,
- target frontier is expanding in a new direction,
- source trust is low or uncertain.

Typical action:

- seed target frontier frame,
- probe highest-disagreement region,
- reduce weight on source saddle and basin terms.

### 11.3 Confirm When

Confirm source basins when:

- confirmation is cheap,
- the basin matters,
- uncertainty about source validity is still high,
- confirmation would allow compute to be spent elsewhere later.

Confirming is not passive sampling. It is an information-gathering action.

### 11.4 Stage-Dependent Weights

Weights can be stage-dependent, but stage should be inferred from evidence, not only iteration number.

Early:

```text
w_confirm   high
w_frontier  moderate
w_saddle    moderate
w_tunnel    low to moderate
w_unvisited low to moderate
```

Middle:

```text
w_saddle    high
w_tunnel    high
w_unvisited high
w_frontier  moderate
w_confirm   lower except for important uncertain basins
```

Late:

```text
w_frontier  high if disagreement remains
w_saddle    high only for unresolved important routes
w_confirm   high only for unresolved important basins
w_unvisited high only for important missing basins
```

### 11.5 Evidence-Driven Weight Adaptation

A cleaner rule:

```text
source_mode_weight = average_trust_in_relevant_region
target_mode_weight = average_disagreement_or_out_of_domain_score
confirm_mode_weight = unresolved_important_region_uncertainty
```

Then:

- if source trust grows, exploit source more,
- if disagreement grows, explore target more,
- if important regions remain untested, confirm or probe them.

### 11.6 Two-Phase Schedule From the Sweep

The sweep (Section 14) shows the explore/exploit tension is real and is the single biggest lever on the curves. The cleanest rule that follows from the data:

```text
fraction_done = sampled_ns / target_budget_ns

if fraction_done < 0.30:
    use FRONTIER-EXPANSION-dominant weights        # discover bins random would miss
elif fraction_done < 0.70:
    use BALANCED weights                            # mix discovery and density-matching
else:
    use DENSITY-MATCHING-dominant weights           # clean up marginal accuracy
```

Why these specific knees:

- **0.30** is roughly where the exploratory variant's coverage curve crosses random's coverage curve in `outputs/compass/exp02`. Up to that point, frontier picks add genuinely new bins faster than i.i.d. random does.
- **0.70** is roughly where the balanced variant's JSD curve drops below random's at the 200 ns budget (`outputs/compass/exp07`). Past that point density-matching begins to actually pay off.

This schedule is evidence-based, not aesthetic. It is the first thing to implement after the sweep, ahead of any new probe types or new source maps. See Sections 14 and 15 for details.

### 11.7 Why Pure Random Is a Hard Baseline (sweep insight)

A classical adaptive-sampling paper would compare against single-trajectory MD. In our retrospective replay, "random" is i.i.d. sampling from the chunk pool, which is i.i.d. from the target marginal. That is a much harder baseline than single-trajectory MD: random replay matches the marginal at the optimal 1/sqrt(n) rate, so JSD-vs-marginal will look indistinguishable from random for any policy that does not explicitly oversample particular bins.

When promoting COMPASS we must avoid claiming wins on a metric where the baseline is provably optimal, and we must explicitly note that real-online sampling is a much weaker baseline than replay-random. Section 18 records this as the headline failure mode.

---

## 12. Full Algorithm

### 12.1 Offline Source Atlas Construction

Inputs:

- long source trajectory,
- source topology,
- source features,
- target-compatible feature mapping.

Steps:

1. Compute source features.
2. Train source TICA and/or SPIB.
3. Project source frames into latent coordinates.
4. Cluster source frames into regions/states.
5. Estimate source density and FES.
6. Estimate source transition graph.
7. Mark transition-like regions and saddles.
8. Estimate mutation sensitivity per region.
9. Define region importance for target scientific goals.
10. Save the source atlas.

Output:

```text
source_atlas = {
  feature_definition,
  latent_model,
  source_states,
  source_density,
  source_fes,
  source_transition_graph,
  source_saddles,
  mutation_sensitivity,
  region_importance
}
```

### 12.2 Target Initialization

Inputs:

- one target starting structure,
- target topology,
- source atlas.

Steps:

1. Compute target features for starting structure.
2. Project starting structure into source atlas.
3. Detect whether it is in-domain or out-of-domain.
4. Assign closest source state.
5. Initialize target belief state.

If the starting structure is deeply out-of-domain, lower source trust globally and start with target-frontier/local-relaxation mode.

### 12.3 First Probe

Run a very short target MD segment, for example 50-100 ps.

This first probe answers:

- does target relax into a source basin?
- does it drift systematically?
- what is local target spread?
- are there immediate boundary crossings?
- what target frames are now eligible as seeds?

### 12.4 Update Step After Every Probe

After each probe:

1. Project new frames into source atlas.
2. Update target density.
3. Update local drift and covariance.
4. Update region trust.
5. Update disagreement score.
6. Detect transitions between source regions.
7. Update route success/failure posteriors.
8. Add new frames to candidate pool.
9. Recompute all candidate scores.

### 12.5 Select Next Seed

Pick the target frame with maximum total score, subject to:

- not too close to the immediately previous seed,
- not in a region that has exceeded a maximum overuse budget,
- not on a route whose failure posterior is already too strong,
- optionally one small randomization among top candidates to avoid deterministic traps.

The optional randomization should be small, for example sample among top 3 candidates by softmax. This preserves deliberate choice while avoiding brittle behavior.

### 12.6 Decide Probe Length

Probe length can also be adaptive:

| Situation | Suggested segment length |
|---|---|
| local relaxation / in-domain check | 50-100 ps |
| saddle test | 100-500 ps |
| promising tunnel | 100-500 ps |
| basin confirmation | 100-300 ps |
| target frontier expansion | 100-500 ps |

Version 1 can use a fixed length for simplicity. Later versions can allow segment length to vary based on expected barrier distance.

### 12.7 Stop Criteria

Stop when one or more are satisfied:

- target observable uncertainty is below threshold,
- important source basins are confirmed, rejected, or visited,
- major source exits are classified open or closed,
- FES estimate stabilizes across recent probes,
- marginal information per segment drops below threshold,
- compute budget is exhausted.

For A8Q2 validation, stop criteria can be evaluated retrospectively against ground truth.

### 12.8 Pseudocode

```text
source_atlas = build_source_atlas(source_md)
belief = initialize_belief(source_atlas, target_start)

trajectory_pool = []
candidate_pool = [target_start]

while compute_budget_remaining:
    scores = {}

    for x in candidate_pool:
        scores[x] = total_score(x, source_atlas, belief)

    seed = argmax_valid(scores)
    tau = choose_probe_length(seed, belief)

    new_traj = run_target_md(seed, tau)
    trajectory_pool.append(new_traj)

    projected = project_to_source_atlas(new_traj, source_atlas)
    belief = update_belief(belief, projected, new_traj)
    candidate_pool = update_candidate_pool(candidate_pool, new_traj, belief)

    if stopping_condition(belief, trajectory_pool):
        break
```

---

## 13. Validation on A10 -> A8Q2

### 13.1 Why A10 -> A8Q2 Is the Right Sandbox

This pair is ideal because:

- A10 long MD exists and can be treated as source prior.
- A8Q2 target ground truth exists and can be hidden during adaptive selection.
- The mutation is small enough that transfer should have some value.
- Existing results already show transfer is partially useful but not sufficient.
- Random adaptive seeding is a strong baseline to beat.

### 13.2 Inputs Already Available

Available on disk:

- A10 features and NHB arrays in `outputs/artifacts/`.
- A8Q2 features and NHB arrays for validation in `outputs/artifacts/`.
- A10 TICA artifacts in `outputs/tica_artifacts_a10/`.
- Transfer summary in `outputs/transfer_a8q2/`.
- Full adaptive strategy metrics in `outputs/full_md_analysis/`.
- A8Q2 seeding outputs in `outputs/seeding_a8q2/`.
- FES figures and grids in `outputs/fes_three_strategies/`.

### 13.3 Validation Mode Without Running New MD

Before spending compute, the algorithm can be replayed offline over existing A8Q2 trajectories:

1. Hide future A8Q2 frames from the decision policy.
2. Start from the known initial target seed.
3. At each decision, select a frame that exists in already generated A8Q2 pools.
4. Reveal the corresponding short segment as if it had just been run.
5. Compare policy choices against random, transferred, and native strategies.

This will not perfectly emulate new MD because the candidate segments are limited to existing data, but it is a cheap sanity check for the scoring logic.

### 13.4 Validation Mode With New Short MD

After offline replay, run a small actual target adaptive test:

- fixed budget, e.g. 5-20 ns total,
- one seed at a time,
- short segments, e.g. 100-500 ps,
- compare against a sequential random baseline of the same aggregate ns.

For a proof of concept, the claim should not be huge speedup. It should be:

> COMPASS finds high-value target regions or reduces FES/observable error faster than sequential random restart under the same compute budget.

---

## 14. BTP Presentation Scope

### 14.1 Why the Full Protocol Is Too Much for the Presentation

The full COMPASS protocol is a research program, not a 4-5 day deliverable. It includes source atlas construction, trust calibration, route posteriors, probe-type balancing, possible SPIB/TICA comparisons, online MD, and rigorous baselines. Trying to prove all of that before a presentation is risky because the bottleneck is not coding; it is execution time and uncertainty about which components will work cleanly.

For the BTP presentation, the goal should be smaller and sharper:

> Show that a source-derived conformational map can make smarter early target-sampling decisions than a naive baseline, and that the idea is promising enough to develop further.

This is enough for a strong "future direction with preliminary result" section, especially because the main project already has completed results from TICA transfer, adaptive seeding, and ratchet-style runs.

### 14.2 BTP Version: COMPASS-Lite

The presentation version should be called **COMPASS-Lite**.

COMPASS-Lite keeps only the components that are cheap to validate:

```text
Score(frame) =
    source_boundary_or_transition_score * trust
  + target_frontier_score * (1 - trust)
  + unvisited_region_score
  + novelty_score
  - redundancy_penalty
  - simple_failure_decay
```

Components included:

- source map from A10 TICA and, if validated, SPIB,
- target-visited frames only,
- target novelty/frontier score,
- lightweight source-target trust score,
- simple failure or overuse decay,
- probe-type logging for interpretability,
- retrospective sequential replay before any new MD.

Components excluded from the BTP version:

- source-structure reservoir insertion,
- MBAR/reweighting claims,
- full Bayesian route posteriors,
- mutation-sensitivity decomposition,
- variable probe lengths,
- committor estimation,
- claims of exact kinetics.

### 14.3 Gated Add-Back Strategy

Because coding is relatively cheap, some components can be added back if they pass cheap diagnostics. Each component must have a kill switch.

| Component | Include if this cheap check passes | Drop if |
|---|---|---|
| TICA map | A10 map separates helix/coil and A8Q2 projects into reasonable support | projection is collapsed or uninformative |
| SPIB map | SPIB latent coordinate correlates with NHB/RMSD or separates metastable states better than TICA | SPIB projection is noisy, collapsed, or hard to explain |
| Trust score | Trust heatmap highlights sensible source-target agreement/disagreement | trust looks random or hurts replay performance |
| Failure decay | Policy stops repeatedly selecting the same stale region | decay suppresses useful high-value regions too early |
| Probe-type logging | Selected seeds can be classified into interpretable reasons | categories are ambiguous or misleading |

This gives a safe path:

1. Start with COMPASS-Lite.
2. Add SPIB only if its diagnostic plot looks useful.
3. Add trust only if it improves or explains choices.
4. Add failure decay only if repeated selections are a problem.
5. Remove any component that does not improve clarity or performance.

### 14.4 Minimum Complete BTP Experiment

The minimum complete experiment should be retrospective first. No new MD should be required for the presentation result.

Steps:

1. Build A10 source map from available artifacts.
2. Build A8Q2 ground-truth reference from existing outputs.
3. Hide future A8Q2 data from the policy.
4. Let COMPASS-Lite choose one seed/chunk at a time from available target data.
5. Compare against random sequential choice and transferred-coordinate-only choice.
6. Plot convergence as a function of revealed target data.

This is not a perfect online simulation, but it is enough to test whether the policy makes intelligent choices over the existing data distribution.

### 14.5 Presentation Graphs

Target graph set:

1. **Source map and chosen probes:** A10 map with COMPASS-Lite selections overlaid, colored by probe type.
2. **JSD vs sampled budget:** COMPASS-Lite vs random vs transferred-coordinate-only.
3. **Coverage vs sampled budget:** how quickly each method covers A8Q2 FES bins.
4. **Helix fraction error vs sampled budget:** whether COMPASS-Lite reaches the observable faster.
5. **Optional score breakdown:** stacked contribution of source, frontier, trust, novelty, and penalty terms for the first few decisions.

If only three figures are ready, prioritize:

1. source/target map with selected probes,
2. JSD or coverage curve,
3. helix-fraction error curve.

### 14.6 Presentation Claims

Safe claim:

> COMPASS-Lite is a proof-of-concept source-prior policy that chooses more interpretable and often more informative early target probes than random restart under the same retrospective budget.

Stronger claim, only if curves support it:

> COMPASS-Lite reaches comparable A8Q2 FES or helix-fraction error with less revealed target sampling than random sequential restart.

Avoid claiming:

- complete solution to mutant MD acceleration,
- universal speedup,
- exact equilibrium recovery,
- exact kinetics,
- source-prior superiority over random in all metrics.

### 14.7 Four-Day Execution Plan

**Day 1: Map and diagnostics**

- Build or load TICA map.
- Check SPIB map if available.
- Generate map plots colored by NHB/RMSD/helix label.
- Decide whether BTP uses TICA-only or TICA+SPIB.

**Day 2: Retrospective policy replay**

- Implement COMPASS-Lite scoring.
- Run sequential replay on existing A8Q2 data.
- Run multiple random baselines.
- Log every selected seed, score component, and probe type.

**Day 3: Figures and interpretation**

- Generate JSD, coverage, and f_helix error curves.
- Generate selected-probe map.
- Generate score/probe-type summary.
- Decide which claims are supported.

**Day 4: Presentation integration**

- Make method slide: "old map + target probes."
- Make result slide: convergence curves.
- Make limitation slide: retrospective proof-of-concept, not full online MD.
- Connect to already completed ratchet and TICA-transfer results.

### 14.8 Optional Real MD Smoke Test

Only run new MD if the retrospective benchmark works and setup is reliable.

Suggested smoke:

- total target budget: 2-5 ns,
- fixed segments: 100-250 ps,
- one seed at a time,
- compare to one random sequential run if time permits.

This smoke test should be a supporting figure, not the main dependency.

### 14.9 How to Position This in the BTP

The BTP story can be:

1. **Completed result:** TICA transfer is imperfect but useful.
2. **Completed result:** Random seeding is a strong baseline for global FES.
3. **Completed/near-completed result:** Ratchet-style runs show targeted acceleration behavior.
4. **New proof-of-concept:** COMPASS-Lite uses the lessons from 1-3 to choose sequential mutant probes more intelligently.

This is scientifically coherent: COMPASS is not a separate unrelated experiment. It is the next method motivated by the observed limitations of direct TICA transfer and naive adaptive seeding.

---

## 15. Metrics and Scientific Claims

### 15.1 Primary Metrics

Use multiple metrics because different methods optimize different things.

| Metric | Meaning |
|---|---|
| JSD vs A8Q2 ground truth FES | Global FES convergence |
| Coverage vs ground truth bins | Exploration completeness |
| f_helix error | Thermodynamic observable accuracy |
| first-passage discovery time | Time to first important transition |
| transition count per ns | Kinetic event discovery |
| route classification accuracy | Did algorithm identify open/closed source routes? |
| trust calibration | Did low-trust regions actually disagree with source? |

### 15.2 Baselines

Compare against:

1. Single long target MD from starting structure.
2. Sequential random restart from target-visited frames.
3. Transferred-TICA seeding.
4. Native-target adaptive seeding, if enough target data exists.
5. Least-counts or frontier-only adaptive sampling.

Random restart must remain a serious baseline because it performed strongly in the existing A8Q2 experiment.

### 15.3 Claims We Can Make If Successful

Strong claim:

> A long source trajectory can be used as a structured prior to reduce target sampling cost, provided the algorithm adaptively learns where the source prior is valid and where it fails.

More specific claim:

> COMPASS reaches the same A8Q2 FES/observable error as random sequential sampling with fewer aggregate target ns.

Mechanistic claim:

> The method accelerates sampling by classifying source basins and routes into confirmed, modified, blocked, and target-novel regions.

### 15.4 Claims We Should Avoid Initially

Avoid claiming:

- universal transfer across unrelated proteins,
- guaranteed 1000x aggregate-compute speedup,
- source TICA is a faithful target reaction coordinate,
- source structures can be inserted without statistical complications,
- short probes alone estimate true target kinetics.

---

## 16. Implementation Plan in This Repository

### 16.1 Proposed New Files

Potential code structure:

```text
src/prior_atlas.py
src/sequential_policy.py
src/trust_update.py
scripts/build_source_atlas_a10.py
scripts/replay_compass_policy.py
scripts/run_compass_a8q2.py
```

### 16.2 Source Atlas Builder

Responsibilities:

- load A10 features,
- load or train TICA/SPIB,
- cluster source states,
- compute density/FES,
- compute transition counts,
- compute mutation sensitivity features,
- save `outputs/compass/source_atlas_a10.json` plus arrays.

### 16.3 Policy Replay Script

Responsibilities:

- simulate sequential decisions using existing A8Q2 data,
- hide future data,
- update belief state step-by-step,
- record selected probe type and score components,
- output learning curves.

This is the lowest-cost development step.

### 16.4 Real MD Runner

Responsibilities:

- choose seed frame,
- extract coordinates,
- assign velocities,
- run short OpenMM segment,
- update candidate pool,
- save all decisions and score components.

All decisions must be logged because interpretability is part of the scientific claim.

### 16.5 Output Artifacts

Suggested outputs:

```text
outputs/compass/
  source_atlas_a10/
  replay_a8q2/
  sequential_run_a8q2/
  decision_log.csv
  score_components.csv
  belief_state_iter_*.json
  curves_jsd_vs_ns.png
  curves_fhelix_error_vs_ns.png
  route_status_summary.csv
```

### 16.6 Minimum Viable Version

The core research MVP should not depend on SPIB, source-structure insertion, or complex Bayesian optimization. For the BTP version, SPIB can still be included as an optional competing map if the cheap diagnostic plots pass.

MVP:

- use A10 TICA/RMSD/NHB source space,
- source bins in 2D or 3D,
- candidate frames from target-visited frames only,
- score using:
  - source transition/boundary value,
  - target novelty,
  - target frontier,
  - source-target disagreement,
  - route failure penalty,
  - redundancy penalty.

Then compare to sequential random.

---

## 17. Failure Modes and Caveats

### 17.1 Source Prior Too Wrong

If the target lies outside source support early, source-guided terms can mislead the algorithm. The remedy is automatic trust reduction and stronger frontier exploration.

### 17.2 Over-Confirmation of Stable Basins

The algorithm may spend too many probes confirming a basin that is already clear. The remedy is to decrease confirmation score rapidly after repeated agreement.

### 17.3 Repeated Saddle Failure

A source saddle may be much harder in target. The algorithm must stop attacking it after enough failed attempts. Route-level Beta posteriors implement this.

### 17.4 Missing New Target States

If the target has a new state not present in source, source-directed terms may not find it. This is why frontier/disagreement terms are required.

### 17.5 Short Probe Misinterpretation

A failed 100 ps saddle probe does not prove a route is closed. It only updates probability. The failure penalty should be gradual, not absolute.

### 17.6 Feature Mapping Problems

Transfer requires equivalent features. Mutations can break side-chain features. Version 1 should emphasize backbone and conserved features, plus mutation-local diagnostics.

### 17.7 Statistical Bias

Adaptive restart sampling is not automatically an equilibrium estimator. For FES and thermodynamics, reweighting or careful comparison to ground truth may be required. The initial scientific objective can be accelerated discovery and convergence diagnostics rather than exact unbiased kinetics.

---

## 18. Open Design Questions

1. Should source states be defined in TICA space, SPIB space, RMSD/NHB space, or a hybrid?
2. How short can a useful probe be for A8Q2: 50 ps, 100 ps, 500 ps?
3. Should the policy pick one deterministic best frame or sample among top candidates?
4. How aggressively should repeated saddle failures reduce route trust?
5. How should mutation sensitivity be computed for short peptides?
6. Should the first version include SPIB, or should SPIB wait until the TICA-based policy is understood?
7. Can offline replay over existing A8Q2 trajectories fairly predict real online performance?
8. When should the no-source-reservoir restriction be relaxed?
9. Which target observable should define "importance": FES JSD, helix fraction, transition discovery, or all of them?
10. How do we present speedup honestly: aggregate ns, wall-clock, or discovery time?

---

## 19. References and Related Ideas

This plan borrows from several existing ideas but combines them differently.

### 19.1 Transferable tICA / MSM for Mutants

Sultan and Pande proposed transferring WT tICA/MSM information to mutants for efficient sampling. Their framing is directly relevant: related systems often share slow modes and metastable states, but basin populations and kinetics change.

Relevance here:

- supports the source-as-prior idea,
- motivates source TICA/MSM atlas,
- warns that transfer assumes similar slow modes.

Difference here:

- this method does not treat transferred coordinates as fixed truth,
- it explicitly updates trust region by region,
- Version 1 avoids direct source-structure insertion.

### 19.2 Adaptive MSM Sampling

Adaptive MSM methods choose new simulations to reduce uncertainty in important states or rates.

Relevance here:

- supports many short targeted simulations,
- motivates uncertainty-aware seed selection,
- provides transition-graph language.

Difference here:

- the initial graph comes from a related source system,
- target compute is used to learn source-target deformation.

### 19.3 LAST, REAP, FAST, kNN-AS

These methods select frontier, outlier, or reward-maximizing structures.

Relevance here:

- target frontier/disagreement term,
- novelty term,
- balance between exploration and exploitation.

Difference here:

- frontier exploration is conditioned by source-map trust,
- repeated failure updates route-level belief.

### 19.4 SPIB and Committor-Like Coordinates

SPIB learns state-predictive low-dimensional representations and can be related to transition-state / committor-like behavior.

Relevance here:

- possible better source atlas than linear TICA,
- useful for identifying uncertain transition regions.

Difference here:

- SPIB is optional in MVP,
- the main algorithm is a sequential decision policy, not only a coordinate-learning method.

### 19.5 Decay-Penalized Prioritized Replay

The idea of prioritizing high-value states but decaying repeatedly failed attempts is highly relevant.

Relevance here:

- route failure penalty,
- Beta posterior over escape probability,
- prevents wasting compute on blocked source exits.

Difference here:

- the decay is tied to physical source routes and target evidence,
- not just replay count.

---

## Working Summary

The full idea can be compressed into one sentence:

> Build an atlas from the long source MD, start from one target structure, and spend each short target MD probe on the frame that best resolves whether to trust, exploit, reject, or ignore a specific part of the source atlas.

The strongest version of the method is not the one that trusts the source most. It is the one that learns fastest where the source is still useful.

