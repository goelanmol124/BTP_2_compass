"""Hints for plugging external WT/mutant benchmarks into COMPASS v2.

Tier-A example: ubiquitin **K11** RADAR/KonDATA — DOI ``10.48606/rx7md621k9zawfz7``.
See ``EXTERNAL_COMPASS_BENCHMARK_SYSTEMS.md`` for links and segment expectations.

Steps (manual): preprocess WT + mutant into ``outputs/artifacts_*`` layout (features + optional NHB),
adapt ``segments`` split per replica boundaries, run::

    python -m src.compass.v2.validate_equivalence --repo-root .
    python -m src.compass.v2.bench --repo-root . --msm-segments "0:T1,T1:T2,..."

until ``build_atlas`` reads your features array paths (extend ``ReplayConfig`` helpers if needed).
""" 
