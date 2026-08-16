"""Deterministic priority scorer for FIRST LIGHT.

priority = round(staleness_h, 3) * round(vulnerable_density, 3) * round(doubt, 3) * (road_cutoff or 1)
Result rounded to 5 decimal places.

No LLM calls. No model access. Pure arithmetic.
"""


def calculate_priority(staleness_h, vulnerable_density, doubt, road_cutoff=None):
    """Compute the FIRST LIGHT rank priority from pre-derived scoring factors.

    staleness_h, vulnerable_density, doubt: numeric factors, each rounded to
    3 decimal places before multiplication.
    road_cutoff: None (clear access, multiplier 1) or a number >= 1.
    """
    if road_cutoff is not None and road_cutoff < 1:
        raise ValueError("road_cutoff must be >= 1 when provided")

    staleness_factor = round(staleness_h, 3)
    vulnerable_density_factor = round(vulnerable_density, 3)
    doubt_factor = round(doubt, 3)
    road_cutoff_factor = road_cutoff if road_cutoff is not None else 1

    priority = staleness_factor * vulnerable_density_factor * doubt_factor * road_cutoff_factor
    return round(priority, 5)
