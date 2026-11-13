from __future__ import annotations
from typing import Dict, Any, Tuple
from RL.config import RL_ENV_CONFIG  


def compute_reward(
        is_repeated_action: bool = False,  
        is_syntax_error: bool = False,  
        db_success: bool = False,  
        is_new_edge: bool = False,  
        complexity_score: float = 0.0,  
        parse_success: bool = False,  
        local_new_edges: int = 0,  
        local_new_nodes: int = 0,  
        structure_operator_count: int = 0,  
        new_operator_count: int = 0,  
        low_freq_operator_count: int = 0,  
        coverage_mode: str = "unrestricted",
        fixed_base_structure: bool = True,
        fixed_base_new_edges: int = 0,
        frontier_new_edges: int = 0,
        frontier_new_nodes: int = 0,
        frontier_mode: bool = False,
        frontier_predicted_new_edges: int = 0,
        frontier_actual_new_edges: int = 0,
        frontier_stagnation_pressure: float = 0.0,
        candidate_bitmap_target_edge_hit: bool = False,
        candidate_bitmap_candidate_level: str = "",
        candidate_bitmap_edge_score: float = 0.0,
        candidate_bitmap_support_rank: int = 0,
        candidate_bitmap_observed_before: bool = False,
        candidate_bitmap_failed_before: bool = False,
        candidate_bitmap_exploration_done: bool = False,
        candidate_bitmap_local_tested: bool = False,
        candidate_bitmap_global_new_edge: bool = False,
        candidate_bitmap_failure_kind: str = "",
        candidate_bitmap_safe_closure_used: bool = False,
        candidate_bitmap_downstream_gap_count: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    reward = 0.0
    reasons = []

    components: Dict[str, float] = {
        "valid_sql": 0.0,
        "parser": 0.0,
        "new_edge": 0.0,
        "frontier_edge": 0.0,
        "old_edge": 0.0,
        "new_operator": 0.0,
        "low_freq_operator": 0.0,
        "frontier_real_new_edge": 0.0,
        "frontier_prediction_hit": 0.0,
        "frontier_prediction_miss": 0.0,
        "candidate_target_hit": 0.0,
        "candidate_target_miss": 0.0,
        "candidate_level": 0.0,
        "candidate_edge_score": 0.0,
        "candidate_support_rank": 0.0,
        "candidate_frontier_explore": 0.0,
        "candidate_local_history": 0.0,
        "candidate_db_invalid": 0.0,
        "candidate_downstream_unavailable": 0.0,
        "execution_error": 0.0,
        "syntax_error": 0.0,
        "repeated": 0.0,
    }

    mode = str(coverage_mode or "unrestricted").lower()
    fixed_base_mode = mode == "fixed_base"

    coverage_info = {
        "coverage_mode": coverage_mode,
        "fixed_base_structure": float(fixed_base_structure),
        "fixed_base_new_edges": int(fixed_base_new_edges),
        "frontier_new_edges": int(frontier_new_edges),
        "frontier_new_nodes": int(frontier_new_nodes),
        "coverage_success_signal": float(
            (int(fixed_base_new_edges) > 0)
            if fixed_base_mode
            else bool(is_new_edge)
        ),
    }

    candidate_level_text = str(candidate_bitmap_candidate_level or "").lower()
    candidate_edge_score_value = max(0.0, min(1.5, float(candidate_bitmap_edge_score or 0.0)))
    candidate_support_rank_value = max(0, int(candidate_bitmap_support_rank or 0))
    candidate_support_rank_norm = max(0.0, min(1.0, candidate_support_rank_value / 60.0))

    candidate_info = {
        "candidate_bitmap_mode": 1.0,
        "candidate_bitmap_target_edge_hit": float(bool(candidate_bitmap_target_edge_hit)),
        "candidate_bitmap_candidate_level": candidate_level_text,
        "candidate_bitmap_edge_score": float(candidate_edge_score_value),
        "candidate_bitmap_support_rank": int(candidate_support_rank_value),
        "candidate_bitmap_support_rank_norm": float(candidate_support_rank_norm),
        "candidate_bitmap_observed_before": float(bool(candidate_bitmap_observed_before)),
        "candidate_bitmap_failed_before": float(bool(candidate_bitmap_failed_before)),
        "candidate_bitmap_exploration_done": float(bool(candidate_bitmap_exploration_done)),
        "candidate_bitmap_local_tested": float(bool(candidate_bitmap_local_tested)),
        "candidate_bitmap_global_new_edge": float(bool(candidate_bitmap_global_new_edge)),
        "candidate_bitmap_failure_kind": str(candidate_bitmap_failure_kind or ""),
        "candidate_bitmap_safe_closure_used": float(bool(candidate_bitmap_safe_closure_used)),
    }

    frontier_mode_flag = bool(frontier_mode)
    frontier_predicted_new_edges_int = max(0, int(frontier_predicted_new_edges or 0))
    frontier_actual_new_edges_int = max(0, int(frontier_actual_new_edges or 0))
    frontier_pressure = max(0.0, min(1.0, float(frontier_stagnation_pressure or 0.0)))

    frontier_prediction_positive_flag = frontier_predicted_new_edges_int > 0
    frontier_prediction_hit_flag = (
        frontier_prediction_positive_flag
        and frontier_actual_new_edges_int > 0
    )
    frontier_prediction_miss_flag = (
        frontier_prediction_positive_flag
        and frontier_actual_new_edges_int <= 0
    )

    frontier_low_freq_scale_applied = False
    frontier_low_freq_scale_value = 1.0
    frontier_low_freq_original_bonus = 0.0
    frontier_low_freq_scaled_bonus = 0.0

    frontier_info = {
        "frontier_mode": float(frontier_mode_flag),
        "frontier_predicted_new_edges": int(frontier_predicted_new_edges_int),
        "frontier_actual_new_edges": int(frontier_actual_new_edges_int),
        "frontier_stagnation_pressure": float(frontier_pressure),
        "frontier_prediction_positive": float(frontier_prediction_positive_flag),
        "frontier_prediction_hit": float(frontier_prediction_hit_flag),
        "frontier_prediction_miss": float(frontier_prediction_miss_flag),
        "frontier_low_freq_scale_applied": 0.0,
        "frontier_low_freq_scale": 1.0,
        "frontier_low_freq_original_bonus": 0.0,
        "frontier_low_freq_scaled_bonus": 0.0,
    }

    failure_kind = str(candidate_bitmap_failure_kind or "").strip().lower()
    syntax_or_placeholder = bool(is_syntax_error) or failure_kind in {
        "syntax_error",
        "placeholder",
        "placeholder_or_hole",
    }
    downstream_unavailable = failure_kind == "downstream_construct_failed"

    if syntax_or_placeholder:
        components["syntax_error"] = float(
            getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_syntax_placeholder_penalty", -5.0)
        )
        reasons.append("Candidate Syntax/Placeholder")
    elif downstream_unavailable:
        components["candidate_downstream_unavailable"] = float(
            getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_downstream_unavailable_penalty", -1.0)
        )
        gap_count = max(0, int(candidate_bitmap_downstream_gap_count))
        if gap_count > 0:
            components["candidate_downstream_unavailable"] = -2.0 * gap_count / (gap_count + 1.0)
        reasons.append("Candidate Downstream Unavailable")
    else:
        if db_success:
            components["valid_sql"] = float(
                getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_valid_sql_bonus", 0.5)
            )
            reasons.append(f"Candidate Valid SQL +{components['valid_sql']:.2f}")

        if parse_success:
            components["parser"] = float(
                getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_parser_success_bonus", 0.5)
            )
            reasons.append(f"Candidate Parser Success +{components['parser']:.2f}")

        if bool(candidate_bitmap_target_edge_hit):
            components["candidate_target_hit"] = float(
                getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_target_hit_bonus", 1.0)
            )
            reasons.append(f"Candidate Target Hit +{components['candidate_target_hit']:.2f}")
        else:
            components["candidate_target_miss"] = float(
                getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_target_miss_penalty", -2.0)
            )
            reasons.append(f"Candidate Target Miss {components['candidate_target_miss']:.2f}")

        if not db_success:
            components["candidate_db_invalid"] = float(
                getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_db_invalid_penalty", -2.0)
            )
            reasons.append(f"Candidate DB Invalid {components['candidate_db_invalid']:.2f}")

        if bool(candidate_bitmap_global_new_edge):
            components["new_edge"] = float(
                getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_global_new_edge_bonus", 10.0)
            )
            reasons.append(f"Global Novel Edge +{components['new_edge']:.2f}")

    if bool(candidate_bitmap_exploration_done):
        components["candidate_local_history"] = float(
            getattr(RL_ENV_CONFIG, "reward_candidate_bitmap_completed_edge_penalty", -5.0)
        )
        reasons.append(
            f"Candidate Globally Completed {components['candidate_local_history']:.2f}"
        )
    elif bool(candidate_bitmap_local_tested) or is_repeated_action:
        components["candidate_local_history"] = float(
            getattr(
                RL_ENV_CONFIG,
                "reward_candidate_bitmap_local_unfinished_retry_penalty",
                -0.25,
            )
        )
        reasons.append(
            f"Candidate Local Unfinished Retry {components['candidate_local_history']:.2f}"
        )

    reward = sum(components.values())
    info: Dict[str, Any] = {
        "reward_total": reward,
        "reward_reason": " + ".join(reasons) if reasons else "Neutral",
        "reward_components": components,
        "reward_valid_sql": components["valid_sql"],
        "reward_parser": components["parser"],
        "reward_new_edge": components["new_edge"],
        "reward_old_edge": components["old_edge"],
        "reward_new_operator": components["new_operator"],
        "reward_low_freq_operator": components["low_freq_operator"],
        "reward_frontier_real_new_edge": components["frontier_real_new_edge"],
        "reward_frontier_prediction_hit": components["frontier_prediction_hit"],
        "reward_frontier_prediction_miss": components["frontier_prediction_miss"],
        "reward_candidate_target_hit": components["candidate_target_hit"],
        "reward_candidate_target_miss": components["candidate_target_miss"],
        "reward_candidate_level": components["candidate_level"],
        "reward_candidate_edge_score": components["candidate_edge_score"],
        "reward_candidate_support_rank": components["candidate_support_rank"],
        "reward_candidate_frontier_explore": components["candidate_frontier_explore"],
        "reward_candidate_local_history": components["candidate_local_history"],
        "reward_candidate_db_invalid": components["candidate_db_invalid"],
        "reward_candidate_downstream_unavailable": components[
            "candidate_downstream_unavailable"
        ],
        "reward_repeated_penalty": components["repeated"],
        "is_repeated_action": float(is_repeated_action),
        "syntax_error": float(is_syntax_error),
        "db_success": float(db_success),
        "parse_success": float(parse_success),
        "is_new_edge": float(bool(candidate_bitmap_global_new_edge)),
        "local_new_edges": int(1 if candidate_bitmap_global_new_edge else 0),
        "local_new_nodes": int(local_new_nodes),
        "structure_operator_count": int(structure_operator_count),
        "new_operator_count": int(new_operator_count),
        "low_freq_operator_count": int(low_freq_operator_count),
        "complexity_score": complexity_score,
        **coverage_info,
        "generation_mode": "candidate_bitmap",
        **frontier_info,
        **candidate_info,
    }
    return float(reward), info


if __name__ == "__main__":
    print("=== Reward Scheme Ready ===")
