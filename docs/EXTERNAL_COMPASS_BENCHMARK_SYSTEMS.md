# External MD systems for COMPASS validation & benchmarks

This note lists **public**, **paired** (or clearly perturbable) molecular dynamics resources that fit the COMPASS setting described in [`CURRENT_PIPELINE.md`](CURRENT_PIPELINE.md) and (historical detail) [`DEPRECATED_COMPASS_PLAN.md`](../archive/docs_deprecated/DEPRECATED_COMPASS_PLAN.md): a **long or rich source** trajectory builds an atlas; a **related target** (mutation, PTM, ligand state, etc.) is explored under a **budget**; **replay** (oracle vs online-faithful / MSM) can be evaluated when enough target data exist.

**Ranking** reflects: explicit **WT vs perturbation** in one place or easy pairing, **replica / segment** structure for honest replay, **documented** integrator and snapshot spacing where known, **open access**, and **practical** size for a student project. **Not** included: resources that are not atomistic MD (e.g. optical-trap force curves mistaken for trajectories).

---

## Tier A — recommended first external tests

### 1. Ubiquitin K11 mutant panel (RADAR / KonDATA)

| Field | Detail |
|--------|--------|
| **Rank** | **1** |
| **Why** | Single deposit: **WT + K11R, K11Q, K11C, K11Ac**; **3 replicas × 200 ns** per variant; ideal **source = wt**, **target = one mutant**; moderate **~781 MB**; **CC0**. |
| **Software / FF** | **GROMACS** (2018 / 2020); **GROMOS 54a7** |
| **Integrator Δt** | Not always stated on landing page; typical **2 fs** for this setup |
| **Trajectory output** | Repository text describes **50 ps** spacing—interpret as **saved-frame interval** (validate in files) |
| **Replay fit** | **900 ns** total per variant (3×200 ns); enough frames for TICA/MSM-style analysis if spacing ≈ 50 ps (~4000 frames / 200 ns) |
| **Caveat** | **GROMOS** vs your **CHARMM36m** peptide sandbox—use **geometry-based** features or document FF mismatch |

**Links**

- DOI: <https://doi.org/10.48606/rx7md621k9zawfz7>
- KonDATA (EN): <https://kondata.uni-konstanz.de/radar/en/dataset/rx7md621k9zawfz7>
- Related article: <https://doi.org/10.1038/s41467-022-33087-1>

---

### 2. CTX-M9 meropenem acyl-enzyme — binding-pocket trajectories (Dryad)

| Field | Detail |
|--------|--------|
| **Rank** | **2** |
| **Why** | **WT + L48A + T165W + S281A**; primary paper reports **>80 ns** per run and **20 simulations** per variant for meropenem acylenzyme; snapshots **every 50 ps**; Dryad bundles **pocket** `.tgz` + **GROMACS `.tpr`**—focused CVs, **CC0**. |
| **Software / FF** | **GROMACS** 4.5 (paper); **AMBER99SB-ILDN**; **TIP3P**; **310 K**, **1 bar** |
| **Trajectory output** | **50 ps** (paper) |
| **Replay fit** | Many independent **~80 ns** legs support **multi-segment** / MSM style replay **if** all runs are in the deposit—**inspect** `*_pocket_trajs.tgz` file count |
| **Caveat** | Pocket subset only—not full-protein observables; F@H/Exacycle mega-campaign in paper is **not** the same as the Dryad pocket bundle |

**Links**

- Dryad: <https://datadryad.org/dataset/doi:10.5061/dryad.ck3dh>
- Article: <https://doi.org/10.1039/C7SC02676E>

---

### 3. SARS-CoV-2 RBD / ACE2 — WT vs N501Y (Mendeley Data)

| Field | Detail |
|--------|--------|
| **Rank** | **3** |
| **Why** | Clear **S/N501Y** comparison; **3 replicas**; **explicit** NAMD protocol on landing page; strong “mutation shifts the map” story |
| **Software / FF** | **NAMD**; **TIP3P**; **PDB 6M0J** |
| **Integrator** | **2 fs** |
| **Production** | **40 ns** per replica |
| **Thermostat** | Langevin, friction **1/5 ps⁻¹**; **310 K** |
| **Cutoff / PME** | **11 Å** nonbonded; **PME grid 1 Å** |
| **Replay fit** | **120 ns** per system if 3×40 ns replicas—enough for **short-lag** probes; **weak** for very slow CVs |
| **Caveat** | **DCD output interval** not on short landing page—read trajectory header / logs after download |

**Links**

- Dataset: <https://data.mendeley.com/datasets/t8f2d5fd4v/1>
- DOI: **10.17632/t8f2d5fd4v.1**

---

### 4. Tryptophan synthase — β Gln114 mutants (Zenodo)

| Field | Detail |
|--------|--------|
| **Rank** | **4 (conditional)** |
| **Why** | **Four enzymatic contexts × (WT + Gln114Ala + Gln114Asn)**; **800 ns** production stated; **AMBER** `parm7` / restarts; excellent multi-mutant design |
| **Caveat** | Public description lists **restarts** and **last-frame PDB**; **confirm** the zip contains **full** trajectory files (`.nc` / `.dcd`) before relying on replay—otherwise rank drops |

**Links**

- Zenodo: <https://zenodo.org/records/10339268>
- DOI: **10.5281/zenodo.10339268**
- Article: <https://doi.org/10.1021/acs.jcim.3c01966>

---

## Tier B — high value with more weight or setup

### 5. FusA:FusC lasso peptide — cyclase complex (GitHub / Nat. Chem. Biol.)

| Field | Detail |
|--------|--------|
| **Rank** | **B1** |
| **Why** | **50 µs aggregate per system**; **100 × 500 ns** segments; **CHARMM36m** + **OpenMM** / **Folding@home**—**matches** your peptide FF narrative; clear **WT vs peptide/cyclase mutants** |
| **Snapshot spacing** | **5000** frames / **500 ns** → **~100 ps** between saved frames |
| **Replay fit** | Designed for **many short legs**; align **MSM segment boundaries** with **500 ns** clones |
| **Caveat** | **~19 GB** per system; heavy I/O |

**Links**

- Repo: <https://github.com/ShuklaGroup/lasso_rotaxane>
- Paper: <https://doi.org/10.1038/s41589-024-01727-w>

---

### 6. Human α-synuclein — WT vs phosphorylation (Zenodo)

| Field | Detail |
|--------|--------|
| **Rank** | **B2** |
| **Why** | **PTM** as perturbation (fits COMPASS §2.2); **GROMACS**; long **µs-scale** equilibrium paths in deposit; multiple force fields may appear—pick one pair for a clean story |
| **Output spacing** | Deposit/README: order **~100 ps** (all-atom) and **~10 ps** (protein+ion subset)—**verify** in README after download |

**Links**

- Zenodo: <https://zenodo.org/records/12605636>
- DOI: **10.5281/zenodo.12605636**

---

### 7. λ Cro — WT vs gain-of-function mutants (Zenodo)

| Field | Detail |
|--------|--------|
| **Rank** | **B3** |
| **Why** | Compact **few-mutation** regulatory system; large consolidated archive |
| **Canonical record** | **10.5281/zenodo.16877369** (record **16877368** redirects here) |
| **Caveat** | **Water-stripped**, **strided** trajectories—match stride to your **chunk** and observables |

**Links**

- Zenodo: <https://zenodo.org/records/16877369>
- Supplemental repo: <https://github.com/WereszczynskiGroup/supplemental-data-hebert-cro-2025>
- bioRxiv: <https://doi.org/10.1101/2025.02.13.638112>

---

### 8. Ubiquitin — Ser65 phosphorylation vs WT (Zenodo)

| Field | Detail |
|--------|--------|
| **Rank** | **B4** |
| **Why** | **`free_md.tar.gz`** for **unbiased** MD; separate **metadynamics** / string-method archives for enhanced sampling—**do not** mix biased and unbiased in the same replay without labeling |
| **Caveat** | Large optional bundles; read **README** inside archive for lengths and saving intervals |

**Links**

- Zenodo: <https://zenodo.org/records/7857674>
- DOI: **10.5281/zenodo.7857674**

---

### 9. Aβ42 + Cu — WT vs Glu22 mutants (Zenodo)

| Field | Detail |
|--------|--------|
| **Rank** | **B5** |
| **Why** | **WT + E22G/E22Q/E22K**; **~2.6 GB** `Cu_mutant_trajectories.tar.gz`; disease-relevant |
| **Caveat** | **PDB-style** trajectory packaging—confirm **atom count**, **solvent**, and **simulation length** in **Heliyon** paper |

**Links**

- Zenodo: <https://zenodo.org/records/2537978>
- Cardiff metadata: <https://research.cardiff.ac.uk/converis/portal/detail/Dataset/123319688?lang=en_GB>
- Article: <https://doi.org/10.1016/j.heliyon.2019.e03071>

---

## Optional scale / prior corpora (not mutation-matched)

Use for **pretraining** or generic dynamics baselines—not as a **single WT/mutant replay pair** without extra work:

| Resource | Role | Link |
|----------|------|------|
| **mdCATH** | Large multi-T MD | <https://doi.org/10.1038/s41597-024-04140-z> · <https://huggingface.co/datasets/compsciencelab/mdCATH> |
| **Chignolin (µs)** | Peptide folding baseline | <https://zenodo.org/records/6349893> |

---

## Explicitly excluded here

| Name | Reason |
|------|--------|
| **Dryad — PCDH15 WT vs V507D** (<https://doi.org/10.5061/dryad.stqjq2c8s>) | Dryad **methods**: **optical-trap force–extension data**, **not** atomistic MD trajectories—wrong modality for COMPASS MD replay |
| **τRAMD / pulling trajectories** (e.g. Zenodo records aimed at **RAMD** dissociation) | **Biased** along a pull axis—different objective than unbiased **prior atlas → target** unless you extend the method |

---

## Suggested testing order in this codebase

1. **Ubiquitin K11** — fastest **multi-replica** mutation panel.  
2. **CTX-M9 pockets** — paper-grounded **50 ps** output; many short runs if bundle is complete.  
3. **Spike N501Y** — short but **clean protocol** text.  
4. **FusA:FusC** — when **statistics + CHARMM36m** matter more than disk.  
5. **Tryptophan synthase** — after **verifying** continuous trajectories inside `md-data.zip`.

---

*Last updated from literature and repository metadata (no trajectory files downloaded for this note). Re-validate URLs and file manifests before production runs.*
