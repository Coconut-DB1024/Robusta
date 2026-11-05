"""
全局配置文件：
- 路径相关（项目根目录、Encoder 模型文件等）
- 强化学习环境参数
- PPO 算法参数
- HNSW 相关配置复用（仅复用 HNSW 的空间与 ef 等参数）
- 消融实验开关配置
- DBMS 配置
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Literal, Tuple







STAGE2_MODE: Literal[
    "base", "enumeration", "representation", "organization", "ppo",
    "representation_organization", "representation_ppo", "organization_ppo", "full",
    "WCL", "Pruning",
] = "full"

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
_MODE_ALIASES = {
    "wcl": ("organization", "WCL"),
    "pruning": ("representation_organization", "Pruning"),
}


def _apply_stage2_mode(mode, *, candidate_config, env_config, algo_config, db_type):
    """Preserve selection controls while retaining aliases in experiment names."""
    requested = str(mode or "").strip().lower()
    normalized, display_name = _MODE_ALIASES.get(requested, (requested, requested))
    if normalized not in _MODE_COMPONENTS:
        raise ValueError(
            f"Unsupported STAGE2_MODE {mode!r}; choose one of "
            + ", ".join((*_MODE_COMPONENTS, *_MODE_ALIASES))
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
    algo_config.experiment_name = f"{str(db_type).strip().lower()}_{display_name}"

    return normalized




try:
    from HNSW import config as hnsw_cfg  
except ImportError:
    hnsw_cfg = None


if hnsw_cfg is not None:
    HNSW_SPACE: str = hnsw_cfg.HNSW_SPACE
    HNSW_M: int = hnsw_cfg.HNSW_M
    HNSW_EF_CONSTRUCTION: int = hnsw_cfg.HNSW_EF_CONSTRUCTION
    HNSW_EF_SEARCH: int = hnsw_cfg.HNSW_EF_SEARCH
else:
    HNSW_SPACE: str = "l2"
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 200





PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


RL_DIR: Path = PROJECT_ROOT / "RL"


DATA_DIR: Path = PROJECT_ROOT / "data"


def get_database_data_path(filename: str) -> Path:
    db_type = str(DB_CONFIG.db_type).strip().lower()
    suffix = {
        "pg": "postgresql",
        "postgres": "postgresql",
        "oracledb": "oracle",
        "oracle_ai26": "oracle",
    }.get(db_type, db_type)
    path = Path(filename)
    return DATA_DIR / f"{path.stem}_{suffix}{path.suffix}"



OUTPUT_DIR: Path = PROJECT_ROOT / "output"


OPERATOR_PROTOTYPES_PATH: Path = OUTPUT_DIR / "operator_prototypes.json"
OPERATOR_SIMILARITY_PATH: Path = OUTPUT_DIR / "operator_similarity.json"
OPERATOR_EMBEDDING_METADATA_PATH: Path = OUTPUT_DIR / "operator_embedding_metadata.json"


CANDIDATE_BITMAP_DIR: Path = OUTPUT_DIR / "candidate_bitmap"
CANDIDATE_EDGE_INDEX_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_index.json"
CANDIDATE_BITMAP_NPZ_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_bitmaps.npz"
CANDIDATE_EDGE_SCORES_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_scores.jsonl"
CANDIDATE_BITMAP_REPORT_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_bitmap_report.json"
CANDIDATE_OBSERVED_REPORT_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_observed_report.json"
CANDIDATE_MISSING_EDGES_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_missing_edges.jsonl"
CANDIDATE_PROBE_GRAPH_PATH: Path = CANDIDATE_BITMAP_DIR / "probe_graph.json"
CANDIDATE_PROBE_REPORT_PATH: Path = CANDIDATE_BITMAP_DIR / "probe_graph_report.json"
CANDIDATE_PROBE_PREDICATE_PATH: Path = CANDIDATE_BITMAP_DIR / "probe_predicate.txt"
CANDIDATE_PROBE_PREDICATE_REPORT_PATH: Path = CANDIDATE_BITMAP_DIR / "probe_predicate_report.json"
CANDIDATE_PROBE_COMPLETED_SQL_PATH: Path = CANDIDATE_BITMAP_DIR / "probe_completed_sql.txt"
CANDIDATE_PROBE_VALIDATION_REPORT_PATH: Path = CANDIDATE_BITMAP_DIR / "probe_validation_report.json"
CANDIDATE_BITMAP_UPDATE_LOG_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_bitmap_update_log.jsonl"
CANDIDATE_BITMAP_UPDATE_REPORT_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_bitmap_update_report.json"


CANDIDATE_EDGE_VECTORS_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_vectors.npy"
CANDIDATE_EDGE_HNSW_INDEX_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_hnsw.bin"
CANDIDATE_EDGE_NEIGHBORS_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_neighbors.npz"
CANDIDATE_EDGE_ORGANIZATION_REPORT_PATH: Path = (
    CANDIDATE_BITMAP_DIR / "candidate_edge_organization_report.json"
)

PREDICATE_KNOWLEDGE_GRAPH_DIR: Path = PROJECT_ROOT / "pkg" / "knowledge_graph"
GLOBAL_KNOWLEDGE_GRAPH_PATH: Path = PREDICATE_KNOWLEDGE_GRAPH_DIR / "global_knowledge_graph.json"


CODET5_DIR: Path = PROJECT_ROOT / "codet5-small"



ENCODER_STATE_PATH: Path = CODET5_DIR / "encoder_state.pt"




@dataclass
class CandidateBitmapConfig:

    enabled: bool = True


    rules_path: Path = PROJECT_ROOT / "pkg" / "configuration" / "postgresql_rules.json"


    bitmap_dir: Path = CANDIDATE_BITMAP_DIR
    edge_index_path: Path = CANDIDATE_EDGE_INDEX_PATH
    bitmap_npz_path: Path = CANDIDATE_BITMAP_NPZ_PATH
    edge_scores_path: Path = CANDIDATE_EDGE_SCORES_PATH
    report_path: Path = CANDIDATE_BITMAP_REPORT_PATH
    observed_report_path: Path = CANDIDATE_OBSERVED_REPORT_PATH
    missing_edges_path: Path = CANDIDATE_MISSING_EDGES_PATH
    probe_graph_path: Path = CANDIDATE_PROBE_GRAPH_PATH
    probe_report_path: Path = CANDIDATE_PROBE_REPORT_PATH
    probe_predicate_path: Path = CANDIDATE_PROBE_PREDICATE_PATH
    probe_predicate_report_path: Path = CANDIDATE_PROBE_PREDICATE_REPORT_PATH
    probe_completed_sql_path: Path = CANDIDATE_PROBE_COMPLETED_SQL_PATH
    probe_validation_report_path: Path = CANDIDATE_PROBE_VALIDATION_REPORT_PATH
    bitmap_update_log_path: Path = CANDIDATE_BITMAP_UPDATE_LOG_PATH
    bitmap_update_report_path: Path = CANDIDATE_BITMAP_UPDATE_REPORT_PATH




    candidate_organization_enabled: bool = True



    candidate_hnsw_enabled: bool = True
    candidate_entropy_enabled: bool = True


    candidate_hnsw_k: int = 32
    candidate_hnsw_min_known_neighbors: int = 4



    candidate_hnsw_priority_weight: float = 1.0
    candidate_entropy_priority_weight: float = 1.0


    candidate_edge_vectors_path: Path = CANDIDATE_EDGE_VECTORS_PATH
    candidate_edge_hnsw_index_path: Path = CANDIDATE_EDGE_HNSW_INDEX_PATH
    candidate_edge_neighbors_path: Path = CANDIDATE_EDGE_NEIGHBORS_PATH
    candidate_edge_organization_report_path: Path = CANDIDATE_EDGE_ORGANIZATION_REPORT_PATH



    probe_predicate_max_depth: int = 3



    global_knowledge_graph_path: Path = GLOBAL_KNOWLEDGE_GRAPH_PATH


    open_strict_initially: bool = True
    open_weak_initially: bool = False
    open_try_initially: bool = False


    include_unknown_as_try: bool = False




    expand_valid_type_terminals: bool = False


    weak_fail_count_to_disable: int = 3
    try_fail_count_to_disable: int = 2
    strict_fail_count_to_disable: int = 5




    target_db_failure_signature_threshold: int = 10




    candidate_bitmap_probe_variant_enabled: bool = True






    candidate_bitmap_probe_variant_max_attempts: int = 35







    candidate_bitmap_shared_reservation_enabled: bool = True




    candidate_bitmap_probe_reservation_timeout_seconds: float = 300.0


    strict_base_score: float = 0.75
    weak_base_score: float = 0.50
    try_base_score: float = 0.30







    candidate_representation_guidance_enabled: bool = True



    pattern_transfer_score_weight: float = 2.00


    refresh_pattern_transfer_before_rl: bool = True




    pattern_transfer_refresh_steps: int = 20000
    pattern_transfer_refresh_min_observed_added: int = 128



    pattern_transfer_top_k_similar_outer: int = 8
    pattern_transfer_min_outer_sim: float = 0.45
    pattern_transfer_max_outer_hop: int = 1
    pattern_transfer_include_self_outer: bool = True






    pattern_transfer_exact_inner_match_score: float = 1.0
    pattern_transfer_same_return_type_match_score: float = 0.65
    pattern_transfer_same_terminal_kind_match_score: float = 0.35
    pattern_transfer_inner_operator_similarity_weight: float = 0.50


    write_human_readable_report: bool = True


CANDIDATE_BITMAP_CONFIG = CandidateBitmapConfig()



@dataclass
class RLEnvConfig: 


    max_steps_per_trajectory: int = 15




    candidate_bitmap_pool_size: int = 64






    candidate_bitmap_enumeration_enabled: bool = False



    candidate_bitmap_enumeration_seed: int = 42








    candidate_bitmap_enumeration_order_mode: str = "cycle_stratified_hard_random"






    candidate_bitmap_enumeration_structural_ratio: float = 0.65




    candidate_bitmap_enumeration_order_group_count: int = 8





    candidate_bitmap_enumeration_independent_worker_order_enabled: bool = True




    candidate_bitmap_enumeration_entropy_bias: float = 0.0




    candidate_bitmap_scored_pool_count: int = 48
    candidate_bitmap_random_pool_count: int = 16



    candidate_bitmap_min_unfinished_per_pool: int = 4



    candidate_bitmap_exploration_done_penalty: float = -1.0



    candidate_bitmap_local_tested_penalty: float = -0.2




    candidate_bitmap_probe_variant_exhausted_threshold: int = 15
    candidate_bitmap_probe_variant_exhausted_penalty: float = -1.5


    candidate_bitmap_probe_variant_exhausted_reset_steps: int = 40000



    candidate_bitmap_inner_ready_priority_weight: float = 1.0







    candidate_bitmap_random_pool_enabled: bool = False









    candidate_bitmap_ppo_selection_enabled: bool = True




    candidate_bitmap_guided_selection_probability: float = 0.70


    candidate_bitmap_guided_selection_temperature: float = 0.50

















    candidate_bitmap_feature_dim: int = 15





    candidate_bitmap_reload_interval_steps: int = 0


    candidate_bitmap_allow_pool_padding: bool = True


    candidate_bitmap_prefer_untested: bool = True




    candidate_bitmap_update_npz_in_env: bool = False



    candidate_bitmap_runtime_probe_dir: Path = CANDIDATE_BITMAP_DIR / "runtime_probe"



    candidate_bitmap_runtime_persist_probe_artifacts: bool = False



    candidate_bitmap_runtime_probe_artifact_sample_every: int = 0



    candidate_bitmap_probe_template_cache_size: int = 5000



    candidate_bitmap_snapshot_save_every_steps: int = 2048




    candidate_bitmap_include_probe_reports_in_info: bool = False



    candidate_bitmap_checkpoint_save_freq: int = 10000


    candidate_bitmap_callback_update_enabled: bool = True



    candidate_bitmap_callback_flush_steps: int = 1024



    candidate_bitmap_callback_min_batch: int = 256



    candidate_bitmap_callback_sync_workers_after_flush: bool = False




    candidate_bitmap_frontier_open_enabled: bool = True



    candidate_bitmap_frontier_open_warmup_steps: int = 500


    candidate_bitmap_frontier_open_weak_pressure: float = 0.50


    candidate_bitmap_frontier_open_try_pressure: float = 0.85



    candidate_bitmap_frontier_weak_min_ratio: float = 0.25



    candidate_bitmap_frontier_try_min_ratio: float = 0.10



    candidate_bitmap_frontier_level_priority_boost: float = 1.20



    candidate_bitmap_frontier_strict_pressure_penalty: float = 0.25






    candidate_bitmap_group_fair_sampling_enabled: bool = True
    candidate_bitmap_group_fair_sampling_warmup_steps: int = 0


    candidate_bitmap_group_fair_target_ratios: tuple = (
        ("core_internal", 0.15),
        ("core_to_extension", 0.15),
        ("extension_outer", 0.70),
    )




    candidate_bitmap_fair_exploration_open_enabled: bool = True
    candidate_bitmap_fair_exploration_warmup_steps: int = 0
    candidate_bitmap_fair_exploration_open_weak: bool = True
    candidate_bitmap_fair_exploration_open_try: bool = True
    candidate_bitmap_fair_exploration_try_warmup_steps: int = 2000

    candidate_bitmap_fair_exploration_weak_min_ratio: float = 0.30
    candidate_bitmap_fair_exploration_try_min_ratio: float = 0.05



    candidate_bitmap_fair_failed_retry_enabled: bool = True
    candidate_bitmap_fair_failed_retry_max_per_pool: int = 6
    candidate_bitmap_fair_failed_retry_max_local_attempts: int = 1
    candidate_bitmap_fair_failed_retry_groups: tuple = (
        "core_internal",
        "core_to_extension",
        "extension_outer",
    )
    candidate_bitmap_fair_failed_retry_error_keywords: tuple = (
        "max_depth",
        "failed to fill",
        "probe predicate construct",
        "probe predicate is empty",
        "construct blocked",
        "VALUE_TYPICAL_",
    )




    candidate_bitmap_core_extension_soft_priority_enabled: bool = False
    candidate_bitmap_core_extension_warmup_steps: int = 2000
    candidate_bitmap_core_extension_activate_on_frontier: bool = True
    candidate_bitmap_core_extension_priority_bonus: float = 0.0
    candidate_bitmap_core_extension_pressure_bonus: float = 0.0
    candidate_bitmap_core_internal_min_ratio: float = 0.34
    candidate_bitmap_core_to_extension_target_ratio: float = 0.33
    candidate_bitmap_extension_outer_max_ratio: float = 0.33
    candidate_bitmap_core_extension_retry_failed_enabled: bool = False
    candidate_bitmap_core_extension_retry_failed_max_per_pool: int = 4
    candidate_bitmap_core_extension_retry_failed_max_local_attempts: int = 1
    candidate_bitmap_core_extension_retry_error_keywords: tuple = (
        "max_depth",
        "failed to fill",
        "probe predicate construct",
        "probe predicate is empty",
        "construct blocked",
    )







    candidate_pool_priority_jitter: float = 0.02




    enable_env_compliance_check: bool = False





    frontier_mode_enabled: bool = True



    frontier_stagnation_window_steps: int = 200



    frontier_min_db_success_in_window: int = 30


    frontier_min_parse_success_in_window: int = 20


    frontier_max_new_edges_in_window: int = 0


    frontier_exit_new_edges_in_window: int = 2










    reward_execution_error: float = -1.0  


    reward_low_freq_operator_threshold: int = 5





    reward_candidate_bitmap_valid_sql_bonus: float = 0.5
    reward_candidate_bitmap_parser_success_bonus: float = 0.5


    reward_candidate_bitmap_global_new_edge_bonus: float = 10.0


    reward_candidate_bitmap_target_hit_bonus: float = 1.0


    reward_candidate_bitmap_target_miss_penalty: float = -2.0


    reward_candidate_bitmap_db_invalid_penalty: float = -2.0
    reward_candidate_bitmap_downstream_unavailable_penalty: float = -1.0


    reward_candidate_bitmap_completed_edge_penalty: float = -5.0
    reward_candidate_bitmap_local_unfinished_retry_penalty: float = -0.25


    reward_candidate_bitmap_syntax_placeholder_penalty: float = -5.0



    reward_candidate_bitmap_edge_score_scale: float = 0.0


    reward_candidate_bitmap_support_rank_scale: float = 0.0


    reward_candidate_bitmap_strict_level_bonus: float = 0.0
    reward_candidate_bitmap_weak_level_bonus: float = 0.0
    reward_candidate_bitmap_try_level_bonus: float = 0.0


    reward_candidate_bitmap_frontier_weak_try_bonus: float = 0.0



    reward_candidate_bitmap_repeated_edge_penalty: float = -0.25


    reward_candidate_bitmap_observed_repeat_penalty: float = -2.0


    reward_candidate_bitmap_failed_before_penalty: float = -1.5



    coverage_mode: str = "fixed_base"















    record_env_step_runtime_stats: bool = False



    stage2_joint_signature_evidence_enabled: bool = True

    stage2_joint_signature_flush_events: int = 5000


    stage2_joint_signature_max_entries: int = 50000




    init_correct_predicates_path: Optional[Path] = field(
        default_factory=lambda: get_database_data_path("Correct_Collection.txt")
    )



    target_sparse_operators: tuple = (
        "LIKE", "IN", "BETWEEN", "IS", "NOT", "AND", "OR", "XOR",
        "!::T_BOOL", "&&::T_BOOL",
        "|::T_CMP", "<<::T_CMP", ">>::T_CMP",
        "ROUND()::T_CMP", "FLOOR()::T_CMP", "LENGTH()::T_CMP",
    )

    high_freq_operators: tuple = (
        "-::T_CMP", ">::T_BOOL", "^::T_CMP", "+::T_CMP",
        "&::T_CMP", "~::T_CMP", ">=::T_BOOL",
    )







@dataclass
class RLAlgoConfig: 


    algo_name: Literal["PPO"] = "PPO"


    learning_rate: float = 3e-4


    gamma: float = 0.99


    n_steps: int = 128


    batch_size: int = 64
    n_epochs: int = 4
    clip_range: float = 0.2
















    n_envs: int = 32


    total_timesteps: int = 400000




    sb3_device: str = "auto"


    seed: int = 42


    progress_log_enabled: bool = True



    progress_log_interval_steps: int = 2048


    experiment_name: str = ""


    progress_log_dir: Path = RL_DIR / "rl_logs" / "ablation_progress"

RL_ALGO_CONFIG = RLAlgoConfig()







def get_target_table(db_type: str, config: Any = None) -> Tuple[str, Optional[str], str]:
    """Return the configured table's catalog name, schema, and SQL identifier."""
    table_name = getattr(config, "table_name", DB_CONFIG.table_name)
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("table_name must be a non-empty string.")
    db_type = str(db_type or "").lower().strip()
    if db_type in {"postgresql", "postgres", "pg"}:
        quoted_table = '"' + table_name.replace('"', '""') + '"'
        return table_name, "public", f'"public".{quoted_table}'
    if db_type in {"oracle", "oracledb", "oracle_ai26"}:

        table_name = table_name.upper()
        quoted_table = '"' + table_name.replace('"', '""') + '"'
        return table_name, None, quoted_table
    if db_type == "sqlserver":
        schema = str(getattr(config, "schema", "dbo") or "dbo")
        quoted_schema = "[" + schema.replace("]", "]]") + "]"
        quoted_table = "[" + table_name.replace("]", "]]") + "]"
        return table_name, schema, f"{quoted_schema}.{quoted_table}"
    quoted_table = "`" + table_name.replace("`", "``") + "`"
    return table_name, None, quoted_table


@dataclass
class DBConfig: 

    db_type: str = "postgresql"


    unix_socket: Optional[str] = None


    host: str = ""
    port: int = 0


    user: str = ""
    password: str = ""


    database: str = ""


    table_name: str = "test_wide_table"


    connect_timeout: int = 5


    query_timeout: int = 10

CONFIGURATION_PATH = PROJECT_ROOT.parent / "configuration.json"


def _read_experiment_configuration():
    config_path = CONFIGURATION_PATH
    try:
        text = config_path.read_text(encoding="utf-8-sig")
        if not text.strip():
            return None
        values = json.loads(text)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"Cannot read experiment configuration: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(values, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return values


_EXPERIMENT_CONFIGURATION = _read_experiment_configuration()


def _load_database_config() -> DBConfig:
    if _EXPERIMENT_CONFIGURATION is None:
        return DBConfig()
    config_path = CONFIGURATION_PATH
    values = dict(_EXPERIMENT_CONFIGURATION)

    values.pop("_comment_db_type", None)
    values.pop("_comment_database", None)
    values.pop("_comment_step", None)
    values.pop("step", None)
    values.pop("_comment_online_model", None)
    values.pop("online_model", None)

    values.setdefault("table_name", DBConfig.table_name)
    expected_fields = {item.name for item in fields(DBConfig)}
    missing = expected_fields - values.keys()
    unknown = values.keys() - expected_fields
    if missing or unknown:
        raise ValueError(
            f"Invalid fields in {config_path}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for name in ("db_type", "host", "user", "password", "database", "table_name"):
        if not isinstance(values[name], str):
            raise ValueError(f"{config_path}: {name} must be a JSON string.")
    if not values["table_name"].strip():
        raise ValueError(f"{config_path}: table_name must not be empty.")
    for name in ("port", "connect_timeout", "query_timeout"):
        if type(values[name]) is not int:
            raise ValueError(f"{config_path}: {name} must be a JSON integer.")
    if values["unix_socket"] is not None and not isinstance(values["unix_socket"], str):
        raise ValueError(f"{config_path}: unix_socket must be a JSON string or null.")
    return DBConfig(**values)


def _load_stage3_sampling_step() -> float:
    value = (_EXPERIMENT_CONFIGURATION or {}).get("step")
    if value is None or (isinstance(value, str) and not value.strip()):
        return 64.0
    if type(value) not in (int, float):
        raise ValueError(f"{CONFIGURATION_PATH}: step must be a positive finite JSON number.")
    try:
        step = float(value)
    except OverflowError:
        raise ValueError(f"{CONFIGURATION_PATH}: step must be a positive finite JSON number.") from None
    if not 0.0 < step < float("inf"):
        raise ValueError(f"{CONFIGURATION_PATH}: step must be a positive finite JSON number.")
    return step


DB_CONFIG = _load_database_config()
RL_ENV_CONFIG = RLEnvConfig()

STAGE2_MODE = _apply_stage2_mode(
    STAGE2_MODE,
    candidate_config=CANDIDATE_BITMAP_CONFIG,
    env_config=RL_ENV_CONFIG,
    algo_config=RL_ALGO_CONFIG,
    db_type=DB_CONFIG.db_type,
)

CANDIDATE_BITMAP_CONFIG.rules_path = (
    PROJECT_ROOT
    / "pkg"
    / "configuration"
    / f"{str(DB_CONFIG.db_type).lower()}_rules.json"
)




@dataclass
class Stage3Config:
    """第三阶段固定深度生成与 WHERE-TLP 验证配置。"""



    stage3_tlp_independent_columns_enabled: bool = True


    stage3_timed_enumeration_enabled: bool = False
    stage3_timed_enumeration_depth: int = 3




    stage3_timed_enumeration_generation_limit_seconds: float = 1800.0
    stage3_timed_enumeration_total_limit_seconds: float = 10800.0




    stage3_timed_random_enabled: bool = False
    stage3_timed_random_depth: int = 3



    stage3_timed_random_generation_limit_seconds: float = 1800.0
    stage3_timed_random_total_limit_seconds: float = 10800.0



    stage3_random_seed: int = 42
    stage3_random_root_pool_size: int = 6

    stage3_random_root_candidate_pool_size: int = 16





    stage3_random_local_window_ratio: float = 0.0
    stage3_random_local_window_min_size: int = 250000



    stage3_random_root_block_size: int = 32768


    stage3_sampling_step: float = _load_stage3_sampling_step()




    stage3_count_memory_budget_mb: int = 256



    stage3_predicate_progress_interval: int = 1000



    stage3_predicate_progress_min_seconds: float = 5.0



    stage3_output_buffer_bytes: int = 1048576



    stage3_compress_predicate_output: bool = True



    stage3_predicate_compression_level: int = 1



    stage3_downstream_reserve_enabled: bool = True


    stage3_downstream_reserve_min_seconds: float = 120.0


    stage3_downstream_reserve_safety_factor: float = 1.25


    stage3_downstream_reserve_fixed_seconds: float = 30.0



    stage3_downstream_reserve_check_interval: int = 1000




    stage3_compress_test_cases: bool = True



    stage3_test_case_compression_level: int = 1



    stage3_report_retention_count: int = 20



    stage3_similarity_order_enabled: bool = True

    stage3_success_guided_sampling_enabled: bool = True

    stage3_candidate_pool_multiplier: int = 2

    stage3_guided_candidate_ratio: float = 0.70




    stage3_evidence_weight_strength: float = 0.25
    stage3_evidence_min_observations: int = 4




    stage3_contextual_null_cast_enabled: bool = True




    stage3_safe_value_guidance_enabled: bool = True
    stage3_safe_value_exploration_ratio: float = 0.01



    stage3_type_projection_enabled: bool = True


    stage3_type_projection_catalog_enabled: bool = True
    stage3_type_projection_evidence_enabled: bool = True


    stage3_type_projection_min_success_count: int = 1




    stage3_type_projection_known_invalid_min_observations: int = 4




    stage3_type_projection_unknown_exploration_ratio: float = 0.01





    stage3_type_projection_max_unknown_variants: int = 1



    stage3_mysql_max_projection_variants: int = 96



    stage3_oracle_max_projection_variants: int = 192







    stage3_mysql_evidence_weight_strength: float = 0.12
    stage3_mysql_value_failure_weight: float = 0.05
    stage3_mysql_execution_failure_weight: float = 0.02
    stage3_mysql_unknown_evidence_factor: float = 0.90






    stage3_oracle_safe_value_exploration_ratio: float = 0.05
    stage3_oracle_implicit_conversion_exploration_ratio: float = 0.05
    stage3_oracle_evidence_weight_strength: float = 0.25
    stage3_oracle_type_failure_weight: float = 1.00
    stage3_oracle_value_failure_weight: float = 0.40
    stage3_oracle_execution_failure_weight: float = 0.20



    stage3_sqlserver_implicit_conversion_exploration_ratio: float = 0.05



    stage3_exact_column_type_guidance_enabled: bool = True


    stage3_column_type_exploration_ratio: float = 0.01







    stage3_runtime_signature_feedback_enabled: bool = True
    stage3_runtime_signature_feedback_min_observations: int = 20
    stage3_runtime_signature_feedback_tiered_enabled: bool = True
    stage3_runtime_signature_feedback_high_rate: float = 0.95
    stage3_runtime_signature_feedback_medium_rate: float = 0.90
    stage3_runtime_signature_feedback_low_rate: float = 0.80
    stage3_runtime_signature_feedback_high_factor: float = 1.80
    stage3_runtime_signature_feedback_medium_factor: float = 0.50
    stage3_runtime_signature_feedback_low_factor: float = 0.10
    stage3_runtime_signature_feedback_very_low_factor: float = 0.01



    stage3_runtime_signature_feedback_strength: float = 0.90
    stage3_runtime_signature_feedback_min_factor: float = 0.10





    stage3_mysql_runtime_type_failure_weight: float = 1.00
    stage3_mysql_runtime_value_failure_weight: float = 0.15
    stage3_mysql_runtime_other_failure_weight: float = 0.05
    stage3_mysql_runtime_timeout_weight: float = 0.10




    stage3_oracle_runtime_type_failure_weight: float = 1.00
    stage3_oracle_runtime_value_failure_weight: float = 0.40
    stage3_oracle_runtime_other_failure_weight: float = 0.20
    stage3_oracle_runtime_timeout_weight: float = 0.05




    stage3_component_signature_feedback_enabled: bool = True
    stage3_component_feedback_trace_stride: int = 16
    stage3_component_feedback_block_size: int = 256
    stage3_component_feedback_max_keys_per_block: int = 32




    stage3_metrics_enabled: bool = True
    stage3_metrics_interval_seconds: float = 600.0




    tlp_error_sample_limit: int = 300


    tlp_error_sql_max_chars: int = 1000


    tlp_error_message_max_chars: int = 500



    tlp_bug_sample_limit: int = 100


    tlp_bug_sql_max_chars: int = 2000



    tlp_progress_log_interval: int = 1000



    tlp_query_timeout: int = 10



    tlp_resource_argument_bounding_enabled: bool = True

    tlp_repeat_count_max: int = 1024


    tlp_repeat_output_max_chars: int = 4096

    tlp_string_length_max: int = 4096


    tlp_resource_bound_sample_limit: int = 5




    tlp_mysql_overflow_retry_enabled: bool = True
    tlp_mysql_exp_argument_limit: float = 700.0



    tlp_pool_chunksize: int = 10


STAGE3_CONFIG = Stage3Config()

_DB_OPERATOR_PROFILE_CACHE = None
_DB_OPERATOR_PROFILE_CACHE_DB_TYPE = None


def load_db_operator_profile(force_reload: bool = False) -> dict:
    global _DB_OPERATOR_PROFILE_CACHE, _DB_OPERATOR_PROFILE_CACHE_DB_TYPE

    db_type = str(DB_CONFIG.db_type)

    if (
        not force_reload
        and _DB_OPERATOR_PROFILE_CACHE is not None
        and _DB_OPERATOR_PROFILE_CACHE_DB_TYPE == db_type
    ):
        return _DB_OPERATOR_PROFILE_CACHE

    default_profile = {
        "target_sparse_operators": list(getattr(RL_ENV_CONFIG, "target_sparse_operators", ())),
        "high_freq_operators": list(getattr(RL_ENV_CONFIG, "high_freq_operators", ())),
        "dirty_generated_tokens": [],
        "bad_function_prefixes": [],
    }

    try:
        profile_path = RL_DIR / "datatype" / f"{db_type}.json"
        if profile_path.exists():
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                profile = data.get("operator_profile", {})
                if isinstance(profile, dict):
                    for key in default_profile:
                        value = profile.get(key)
                        if isinstance(value, list):
                            default_profile[key] = [str(x) for x in value]
    except Exception:
        pass

    _DB_OPERATOR_PROFILE_CACHE = default_profile
    _DB_OPERATOR_PROFILE_CACHE_DB_TYPE = db_type
    return _DB_OPERATOR_PROFILE_CACHE

_DB_COVERAGE_PROFILE_CACHE = None
_DB_COVERAGE_PROFILE_CACHE_DB_TYPE = None


def load_db_coverage_profile(force_reload: bool = False) -> dict:
    """
    读取当前数据库的 coverage profile。

    优先读取：
        RL/datatype/<db_type>_coverage_profile.json

    返回字段：
        base_node_ids
        fixed_base_target_sparse_operators

    如果文件不存在或损坏，返回空 profile。
    """
    global _DB_COVERAGE_PROFILE_CACHE, _DB_COVERAGE_PROFILE_CACHE_DB_TYPE

    db_type = str(DB_CONFIG.db_type)

    if (
        not force_reload
        and _DB_COVERAGE_PROFILE_CACHE is not None
        and _DB_COVERAGE_PROFILE_CACHE_DB_TYPE == db_type
    ):
        return _DB_COVERAGE_PROFILE_CACHE

    default_profile = {
        "base_node_ids": [],
        "fixed_base_target_sparse_operators": [],
    }

    try:
        profile_path = RL_DIR / "datatype" / f"{db_type}_coverage_profile.json"
        if profile_path.exists():
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for key in default_profile:
                    value = data.get(key)
                    if isinstance(value, list):
                        default_profile[key] = [str(x) for x in value]
    except Exception:
        pass

    _DB_COVERAGE_PROFILE_CACHE = default_profile
    _DB_COVERAGE_PROFILE_CACHE_DB_TYPE = db_type
    return _DB_COVERAGE_PROFILE_CACHE






@dataclass
class MonitorConfig:

    monitor_log_path: Path = OUTPUT_DIR / "monitor_log.csv"
    """
    # 最终全局正确谓词的持久化文本路径（代替原来的 init_correct_predicates_path 直接追加写）
    global_correct_txt_path: Path = OUTPUT_DIR / "Global_Correct_Collection.txt"
    """

    global_correct_txt_path: Path = RL_ENV_CONFIG.init_correct_predicates_path


    global_pkg_pkl_path: Path = OUTPUT_DIR / "global_pkg.pkl"


    io_batch_size: int = 300













    pkg_parse_workers: int = 4


    pkg_parse_min_batch: int = 64



MONITOR_CONFIG = MonitorConfig()


@dataclass
class RecycleConfig:
    """
    阶段 6C：RL 训练结束后的谓词回炉配置。

    注意：
        这里不负责第一阶段重训；
        这里只控制是否在 RL 结束后调用 RL/predicate_recycle.py
        生成 recycle_candidates.json / recycle_apply_log.json。
    """


    enable_after_training: bool = True





    recycle_enable_periodic_trigger: bool = True








    recycle_interval_steps: int = 80000



    recycle_stop_training_after_trigger: bool = True



    recycle_run_after_periodic_stop: bool = True



    recycle_top_k: int = 1000


    recycle_min_score: float = 1.0


    recycle_include_initial: bool = False




    recycle_require_new_contribution: bool = True



    recycle_apply_after_training: bool = False


    recycle_apply_top_k: int = 500






    recycle_dry_run_after_training: bool = True


    recycle_create_backup: bool = True

    recycle_abstract_cap: int = 5
    recycle_apply_require_value_signal: bool = True
    recycle_disallow_high_freq_zero_signal: bool = True




    recycle_frontier_new_edges_as_strong: bool = True



    recycle_allow_frontier_non_fixed_base_apply: bool = True



    recycle_low_freq_structure_in_value_channel: bool = False



    recycle_low_freq_structure_in_novel_channel: bool = True


    recycle_fixed_base_new_edges_score_weight: float = 5.0



    recycle_frontier_new_edges_score_weight: float = 5.0



    recycle_enable_novel_structure_fill: bool = True
    recycle_novel_structure_target_count: int = 120
    recycle_fixed_base_require_base_signal: bool = True

    recycle_strong_target_quota: int = 250
    recycle_sparse_target_quota: int = 150
    recycle_diverse_target_quota: int = 75
    recycle_stable_target_quota: int = 25


RECYCLE_CONFIG = RecycleConfig()



@dataclass
class AutoCycleConfig:
    """
    阶段 6F / 6D-2：自动循环编排配置。

    一个 auto cycle 的基本流程是：
        1. python main.py rl_cycle
        2. python -m RL.predicate_recycle --apply ...
        3. python main.py recycle_pre_rl

    说明：
        这里不直接实现训练逻辑，只控制 main.py auto_cycle 如何用子进程串联已有入口。
    """









    auto_cycle_auto_compute_num_cycles: bool = True






    auto_cycle_num_cycles: int = 3



    auto_cycle_run_pre_rl_if_missing: bool = True


    auto_cycle_recycle_top_k: int = 1000


    auto_cycle_recycle_min_score: float = 1.0



    auto_cycle_recycle_apply_top_k: int = 500








    auto_cycle_recycle_dynamic_schedule_enabled: bool = True



    auto_cycle_recycle_value_novel_schedule: tuple = (
        (700, 240),
        (600, 200),
        (500, 160),
    )


    auto_cycle_recycle_default_value_after_schedule: int = 400
    auto_cycle_recycle_default_novel_after_schedule: int = 120



    auto_cycle_recycle_include_initial: bool = False



    auto_cycle_recycle_allow_no_new_contribution: bool = False




    auto_cycle_recycle_dry_run: bool = False



    auto_cycle_recycle_no_backup: bool = False



    auto_cycle_skip_retrain_if_no_apply: bool = True



    auto_cycle_run_recycle_pre_rl: bool = True



    auto_cycle_stop_on_failure: bool = True












    auto_cycle_run_stage3_after_cycles: bool = True




    auto_cycle_verify_operator_similarity_after_retrain: bool = True


AUTO_CYCLE_CONFIG = AutoCycleConfig()


