"""Apply Stage-2 experiment modes to the existing selection controls."""


_MODE_COMPONENTS = {
    "base": (False, False, False),
    "enumeration": (False, False, False),
    "representation": (True, False, False),
    "organization": (False, True, False),
    "ppo": (False, False, True),
    "representation_organization": (True, True, False),
    "representation_ppo": (True, False, True),
    "organization_ppo": (False, True, True),
    "full": (True, True, True),
}


def apply_stage2_mode(mode, *, candidate_config, env_config, algo_config, db_type):
    """Set only experiment selection controls; preserve graph and training settings."""
    normalized = str(mode or "").strip().lower()
    if normalized not in _MODE_COMPONENTS:
        raise ValueError(
            f"Unsupported STAGE2_MODE {mode!r}; choose one of "
            + ", ".join(_MODE_COMPONENTS)
        )

    representation, organization, ppo = _MODE_COMPONENTS[normalized]
    candidate_config.candidate_representation_guidance_enabled = representation
    candidate_config.candidate_organization_enabled = organization
    candidate_config.candidate_hnsw_enabled = organization
    candidate_config.candidate_entropy_enabled = organization
    env_config.candidate_bitmap_ppo_selection_enabled = ppo
    env_config.candidate_bitmap_random_pool_enabled = (
        not (representation or organization) and normalized != "enumeration"
    )
    env_config.candidate_bitmap_enumeration_enabled = normalized == "enumeration"
    algo_config.experiment_name = f"{str(db_type).strip().lower()}_{normalized}"
    return normalized
