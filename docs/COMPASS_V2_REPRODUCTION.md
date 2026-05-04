# COMPASS v2 headline reproduction (A10 → A8Q2)

Equivalent **intent** to BTP Figures for TICA vs SPIB, under the **v2 causal replay** abstraction (numerics differ from legacy oracle replay).

From repo root:

```bash
python scripts/compass_v2_validate_equivalence.py --methods M3,M5,M7,M8 --anchors 200
python scripts/compass_v2_bench.py --build-eq-only --eq-method M3 --eq-k 40

python scripts/compass_v2_bench.py \
  --map-kind tica2d --schedule two_phase --deficit-mode information_gain \
  --policy compass --chunk-frames 10 --n-steps 80 --eq-method M3 --eq-k 40 \
  --output-dir outputs/v2/e08_style_tica2d --run-random-summary

python scripts/compass_v2_bench.py \
  --map-kind spib1d --schedule two_phase \
  --policy compass --chunk-frames 10 --n-steps 80 --eq-method M3 --eq-k 40 \
  --output-dir outputs/v2/e06_style_spib1d --run-random-summary
```

Structural **M1** / **M8** require A8Q2 DCDs under `outputs/btp_equil/A8Q2/`.
