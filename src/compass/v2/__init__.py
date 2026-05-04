"""COMPASS v2 — seed-selection benchmark with equivalence-cloud candidates.

Primary headline benchmark for: *given source MD, does a smarter restart policy*
*accelerate target sampling versus random-from-the-same-online-plausible-candidate-set?*

Run ``python -m src.compass.v2.bench --help`` or see ``COMPASS_PLAN.md`` §1.2.
"""

from src.compass.v2.equivalence import EquivalenceMethod, apply_gates

__all__ = ["EquivalenceMethod", "apply_gates"]
