# Output directories after cleanup

## Active (`outputs/` — still in place)

| Path | Role |
|------|------|
| `outputs/btp_equil/` | A10 / A8Q2 equilibration and production MD |
| `outputs/phase5_a10/`, `outputs/phase5_a8q2/` | Phase-5 peptide diagnostics |
| `outputs/full_md_analysis/`, `outputs/deep_analysis/` | Longer MD / seeding analyses |
| `outputs/transfer_a8q2/`, `outputs/tica_artifacts_a10/`, `outputs/artifacts/` | Transfer / feature caches |
| `outputs/v2/` | COMPASS v2 benchmarks and oracle artifacts |
| `outputs/runpod_sync/` | Sync manifests / metadata |

## Archived (`archive/outputs_legacy/` — moved)

Former siblings of the paths above: all COMPASS sweep folders (`compass`, `compass_sweep2`, `compass_sweep3`, `compass_sweep_online_*`, …), MSM benchmark trees, ratchet demos, seeding batches, SPIB model/analysis outputs, TICA sweep explorations, FES figures roots, smoke pipelines, etc.

Any script or markdown that still mentions `outputs/compass_sweep2/` or `outputs/msm_benchmark/` for **completed** runs should be read as **`archive/outputs_legacy/<same-relative-path>`** unless you rerun experiments into fresh `outputs/…` directories.
