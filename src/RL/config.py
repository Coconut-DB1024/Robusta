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
from typing import Optional, Literal

# 第二阶段统一模式；下方表示学习、组织、PPO、候选池和枚举字段由此自动设置。
# representation / organization / ppo 分别表示仅启用对应的选边机制。
# 双机制使用 representation_organization / representation_ppo / organization_ppo。
# base 使用纯随机选边；enumeration 使用现有枚举流程；full 启用全部三项。
# 实验名称也由模式自动生成。
STAGE2_MODE: Literal[
    "base", "enumeration", "representation", "organization", "ppo",
    "representation_organization", "representation_ppo", "organization_ppo", "full",
] = "base"

# HNSW 相关配置复用

try:
    from HNSW import config as hnsw_cfg  # 假定 HNSW 是项目根目录下的一个包
except ImportError:
    hnsw_cfg = None

# 如果 HNSW.config 存在，则复用其中的参数；否则给出合理的默认值（方便单独调试 RL）
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


# ====================== 1. 路径相关配置 ======================

# 当前文件: 项目根目录/RL/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# RL 目录本身
RL_DIR: Path = PROJECT_ROOT / "RL"

# 存放原始谓词文本的 data 目录
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


# HNSW 索引输出目录
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# 第一阶段新增的算子表示与相似算子表输出路径
OPERATOR_PROTOTYPES_PATH: Path = OUTPUT_DIR / "operator_prototypes.json"
OPERATOR_SIMILARITY_PATH: Path = OUTPUT_DIR / "operator_similarity.json"
OPERATOR_EMBEDDING_METADATA_PATH: Path = OUTPUT_DIR / "operator_embedding_metadata.json"

# 阶段17-1：Bitmap Candidate Map 输出路径
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

# 候选边表示、HNSW 近邻与结构熵的输出路径。
CANDIDATE_EDGE_VECTORS_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_vectors.npy"
CANDIDATE_EDGE_HNSW_INDEX_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_hnsw.bin"
CANDIDATE_EDGE_NEIGHBORS_PATH: Path = CANDIDATE_BITMAP_DIR / "candidate_edge_neighbors.npz"
CANDIDATE_EDGE_ORGANIZATION_REPORT_PATH: Path = (
    CANDIDATE_BITMAP_DIR / "candidate_edge_organization_report.json"
)

PREDICATE_KNOWLEDGE_GRAPH_DIR: Path = PROJECT_ROOT / "pkg" / "knowledge_graph"
GLOBAL_KNOWLEDGE_GRAPH_PATH: Path = PREDICATE_KNOWLEDGE_GRAPH_DIR / "global_knowledge_graph.json"

# 第一阶段 CodeT5 Encoder 模型所在目录。
CODET5_DIR: Path = PROJECT_ROOT / "codet5-small"


# active 第一阶段只保存 Encoder 与共享 token embedding。
ENCODER_STATE_PATH: Path = CODET5_DIR / "encoder_state.pt"


# ====================== 2.5 Bitmap Candidate Map 配置 ======================

@dataclass
class CandidateBitmapConfig:
    # 是否启用阶段17-1候选依赖边全集构建。
    enabled: bool = True

    # DBMS 规则文件路径。
    rules_path: Path = PROJECT_ROOT / "pkg" / "configuration" / "postgresql_rules.json"

    # 输出路径。
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

    # ================== 候选边已知集合组织（HNSW + 结构熵） ==================
    # 总开关：True 时按下方两个独立开关启用 HNSW / 结构熵；
    # False 时两项组织信号均不参与候选边排序，并保持中性特征值。
    candidate_organization_enabled: bool = True

    # 两项独立开关用于调试：True 表示启用对应组织信号，False 表示禁用；
    # 正式“集合组织”消融时建议同时开启或同时关闭。
    candidate_hnsw_enabled: bool = True
    candidate_entropy_enabled: bool = True

    # 每条候选边预计算的相似邻居数量，以及计算成功倾向所需的最少已知邻居数。
    candidate_hnsw_k: int = 32
    candidate_hnsw_min_known_neighbors: int = 4

    # 加入原候选优先级的权重。HNSW 倾向以 0.5 为中性点；
    # 结构熵惩罚会随 frontier 停滞压力升高而自动减弱。
    candidate_hnsw_priority_weight: float = 1.0
    candidate_entropy_priority_weight: float = 1.0

    # 候选边向量、HNSW 索引、预计算邻居和可读报告的保存位置。
    candidate_edge_vectors_path: Path = CANDIDATE_EDGE_VECTORS_PATH
    candidate_edge_hnsw_index_path: Path = CANDIDATE_EDGE_HNSW_INDEX_PATH
    candidate_edge_neighbors_path: Path = CANDIDATE_EDGE_NEIGHBORS_PATH
    candidate_edge_organization_report_path: Path = CANDIDATE_EDGE_ORGANIZATION_REPORT_PATH
    # ======================================================================

    # probe predicate 构造最大递归深度，防止 AND -> AND、NOT -> NOT 这类自环无限展开。
    probe_predicate_max_depth: int = 3

    # 注意：global_knowledge_graph.json 位于 pkg/knowledge_graph，
    # 不在 output 目录。
    global_knowledge_graph_path: Path = GLOBAL_KNOWLEDGE_GRAPH_PATH

    # 初始阶段只开放 strict；weak / try 在后续 frontier 阶段逐步开放。
    open_strict_initially: bool = True
    open_weak_initially: bool = False
    open_try_initially: bool = False

    # 是否把规则无法判断的组合放入 try_candidate_bitmap。
    include_unknown_as_try: bool = False

    # 是否根据 db rules JSON 的 valid_types 自动扩展 COL_*/VALUE_* 终端节点。
    # 默认 False：阶段17-1先只使用 candidate_bitmap.terminal_tokens 中显式声明的终端节点，
    # 避免 COL_BIGINT / COL_JSON / COL_BLOB 等当前 schema 中不存在的节点放大候选全集。
    expand_valid_type_terminals: bool = False

    # 失败处理：失败不是一次永久删除，而是先累计 fail_count。
    weak_fail_count_to_disable: int = 3
    try_fail_count_to_disable: int = 2
    strict_fail_count_to_disable: int = 5

    # Candidate bitmap 失败判定阈值 N：
    # 只有“DB 执行失败 + parser 解析成功 + target edge 命中”的 probe 才能作为目标边失败证据。
    # 同一条 edge_id 下，相同 predicate_template 只计一次；累计 N 个不同失败谓词后才写入 failed_bitmap。
    target_db_failure_signature_threshold: int = 10

    # Candidate bitmap probe 构造变体：
    # enabled=True 时，同一条目标边会记住已经尝试过的 predicate_template，
    # 下次构造会固定目标 slot，只切换非目标 slot 的候选组合，避免反复测试同一个谓词。
    candidate_bitmap_probe_variant_enabled: bool = True
    # 单次构造最多尝试多少个非目标 slot 组合来寻找“没试过”的 predicate。
    # 这个值不是失败判定 N；N 控制多少个不同 DB 失败证据才写 failed_bitmap。
    # 单次构造最多尝试 35 个非目标 slot 组合；失败判定阈值仍保持 N=10。
    # 该参数同时作用于 base 和 full，因为两者使用同一套 probe 构造流程。
    # Maximum non-target-slot variants attempted for one edge in one step;
    # this applies to both base and full and does not change the failure threshold N=10.
    candidate_bitmap_probe_variant_max_attempts: int = 35

    # PPO 最终目标边的轻量共享预占：True 表示仅在 PPO 64 选 1 后原子预占该1条边；
    # False 表示不进行跨环境预占。该开关不再预占整个64条候选池。
    # base（candidate_bitmap_ppo_selection_enabled=False）始终不使用分片和预占，
    # 保留全图随机选择以及不同环境可能选中同一条边的基线语义。
    # 预占状态不写 candidate_bitmaps.npz；callback 更新内存共享状态后立即释放，
    # 正式 bitmap 仍按原频率集中批量回写。
    candidate_bitmap_shared_reservation_enabled: bool = True

    # 单边共享预占超时（秒）。环境异常退出或 callback 未收到结果时自动释放，
    # 防止某条目标边永久处于 pending 状态；正常结果到达 callback 后立即释放。
    # Maximum lifetime of a pending reservation before it is reclaimed (seconds).
    candidate_bitmap_probe_reservation_timeout_seconds: float = 300.0

    # edge_score 的基础权重。
    strict_base_score: float = 0.75
    weak_base_score: float = 0.50
    try_base_score: float = 0.30

    # 阶段18：Transformer 相似度不再用于 sim(outer, inner) 直接连边打分。
    # 新逻辑是 slot pattern transfer：
    #   sim(current_outer, observed_outer)
    #   -> 迁移 observed_outer 同 slot 已验证过的 inner 接受模式。
    # True：上述表示迁移评分参与第二阶段选边；False：该评分按 0 处理。
    # False 不关闭第一阶段，HNSW 与结构熵也继续由各自开关独立控制。
    candidate_representation_guidance_enabled: bool = True

    # 已有 pattern_transfer_score 在第二阶段候选排序中的放大系数；
    # 不改变第一阶段编码或原始相似度，只提高表示学习对选边的影响。
    pattern_transfer_score_weight: float = 2.00

    # 每轮 RL 开始前强制刷新一次 pattern_transfer_score。
    refresh_pattern_transfer_before_rl: bool = True

    # RL 运行中低频刷新条件：
    # 距离上次刷新超过 pattern_transfer_refresh_steps，
    # 且期间新增 observed edge 数达到 pattern_transfer_refresh_min_observed_added。
    pattern_transfer_refresh_steps: int = 20000
    pattern_transfer_refresh_min_observed_added: int = 128

    # 阶段18-2：slot pattern transfer 细粒度参数。
    # 查找与 current_outer 相似的 observed_outer 时使用。
    pattern_transfer_top_k_similar_outer: int = 8
    pattern_transfer_min_outer_sim: float = 0.45
    pattern_transfer_max_outer_hop: int = 1
    pattern_transfer_include_self_outer: bool = True

    # inner 节点匹配分数：
    # exact: I == J；
    # same_return_type: return_type 相同；
    # same_terminal_kind: 同为 COL_* / VALUE_* / literal 等终端类别；
    # inner_operator_similarity_weight: I 和 J 都是算子时，使用 sim(I, J) 的权重。
    pattern_transfer_exact_inner_match_score: float = 1.0
    pattern_transfer_same_return_type_match_score: float = 0.65
    pattern_transfer_same_terminal_kind_match_score: float = 0.35
    pattern_transfer_inner_operator_similarity_weight: float = 0.50

    # 是否保存可读报告，便于先 dry-run 检查。
    write_human_readable_report: bool = True


CANDIDATE_BITMAP_CONFIG = CandidateBitmapConfig()

# ====================== 3. 强化学习环境配置 ======================

@dataclass
class RLEnvConfig: # 每步选择一条候选边，构造谓词并验证 SQL。

    # 一条 trajectory 中最多允许多少步（多少个谓词）
    max_steps_per_trajectory: int = 15


    # candidate_bitmap 模式下，每个 step 给 PPO 看的候选边数量。
    # action_space = Discrete(candidate_bitmap_pool_size)
    candidate_bitmap_pool_size: int = 64

    # candidate_bitmap 暴力枚举消融开关，优先级高于 PPO、随机 Base 和所有评分引导：
    # True：按下方 order_mode 执行枚举，不构建 48+16 候选池，
    #       不读取规则层级、表示学习、HNSW 或 PPO action；新模式每个 worker 每轮刷新顺序。
    # False：完全保留现有 PPO / Base / 评分引导选边流程。
    # 枚举模式只替换目标边选择；谓词构造、DB 执行、Bitmap、正式图、回流和重训保持不变。
    candidate_bitmap_enumeration_enabled: bool = False

    # 枚举顺序的基础种子。cycle_stratified_hard_random 会同时混入 RL 轮次和 worker_rank，
    # 使每个环境在每轮刷新自身顺序；其余旧枚举模式继续保持原有固定种子语义。
    candidate_bitmap_enumeration_seed: int = 42

    # 仅在枚举总开关为 True 时生效：
    # "cycle_stratified_hard_random"：每个 worker 每轮独立生成完整无重复排列；每轮本 worker
    #                                      的预计执行步数内按下方比例交织结构复杂边与随机边，
    #                                      结构复杂边内部按多参数、偏上游程度优先排列。
    # "top_down_operator_block"：按外层算子分块，参数位更多、非终端依赖比例更高的块优先；
    #                              不读取结构熵、评分或任何成功/失败结果。
    # "independent_random"：保留每个 worker 独立固定随机排列的旧枚举方式。
    candidate_bitmap_enumeration_order_mode: str = "cycle_stratified_hard_random"

    # 仅在 cycle_stratified_hard_random 枚举中生效：每个 worker 每轮预计执行步数中，
    # 结构复杂边所占比例；0.65 表示预计执行前缀中 65% 为结构复杂边、35% 为随机边。
    # 当前每轮 80000 步、32 个 worker 时，每个 worker 的 2500 步前缀对应
    # 1625 条结构复杂边和 875 条随机边；实际步数仍由每轮总步数和 worker 数动态计算。
    # 结构复杂仅由静态候选元数据判断，不读取评分、Bitmap 或任何执行结果。
    candidate_bitmap_enumeration_structural_ratio: float = 0.65

    # 仅在 top_down_operator_block 枚举中生效：把全部环境划成多少个算子块顺序组。
    # 当前 32 个环境分为 8 组、每组 4 个；同组共享算子块顺序，但块内边顺序各自固定随机。
    # 该参数只改变枚举模式的跨环境覆盖方式，不改变 PPO、Base 或其他消融实验。
    candidate_bitmap_enumeration_order_group_count: int = 8

    # 仅在旧枚举顺序模式中生效；cycle_stratified_hard_random 固定为每个 worker 独立顺序：
    # True：每个 worker 使用 hash(seed, worker_rank, edge_id) 生成独立固定排列，
    #       允许不同 worker 自然重复选边，不使用共享游标、锁或预占。
    # False：保留32个环境按 worker_rank 全局无重复分批领取边的协调枚举。
    candidate_bitmap_enumeration_independent_worker_order_enabled: bool = True

    # 仅在协调枚举模式下生效：按 structural_entropy 从高到低排序的优先区占比。
    # 0.0 表示纯固定随机顺序；独立 worker 枚举固定忽略该参数并保持纯随机。
    # 该参数不启用候选组织/HNSW，也不影响 PPO、Base 或其他评分引导模式。
    candidate_bitmap_enumeration_entropy_bias: float = 0.0

    # 非纯随机模式下，64 条候选边由“48 条评分边 + 16 条随机边”组成：
    # 评分部分保留表示学习、HNSW/熵、层级与 edge group 等现有评分/配额机制；
    # 随机部分用于保持探索多样性，避免高分旧边长期占满候选池。
    candidate_bitmap_scored_pool_count: int = 48
    candidate_bitmap_random_pool_count: int = 16

    # 非纯随机模式下，每个候选池尽量至少保留的“尚未探索结束”边数。
    # 若全局剩余未结束边不足该数量，则全部纳入，不会报错或重复造边。
    candidate_bitmap_min_unfinished_per_pool: int = 4

    # observed_bitmap 或 confirmed failed_bitmap 为 1 时统一视为 exploration_done；
    # 该值只用于候选排序软降权，不从候选全集硬删除；PPO 奖励使用下方独立的 -5 参数。
    candidate_bitmap_exploration_done_penalty: float = -1.0

    # 当前 env 已经测试过该边时的候选排序轻微软降权；不会把尚未确认失败的边排除。
    # PPO 对本地未结束重试的奖励使用下方独立的 -0.25 参数。
    candidate_bitmap_local_tested_penalty: float = -0.2

    # probe_variant_exhausted 的窗口化软降权只作用于 candidate_bitmap 候选排序：
    # 单个 env 在当前窗口内对同一条边累计达到 15 次后，优先级固定扣 1.5；
    # 不增加 fail_count、不写 failed_bitmap，也不改变全局 N=10 的 DB 失败判定。
    candidate_bitmap_probe_variant_exhausted_threshold: int = 15
    candidate_bitmap_probe_variant_exhausted_penalty: float = -1.5
    # 该窗口使用全部同步环境合计的全局 RL 步数；40000 表示每半轮 80000 步重置一次。
    # 计数由各 env 本地维护，不跨进程共享，因此不会引入锁或同步开销。
    candidate_bitmap_probe_variant_exhausted_reset_steps: int = 40000

    # inner_ready=1 表示目标 inner 可完全沿正式图已有关系构造。
    # 该奖励只参与非纯随机候选排序；纯随机 base 不读取该值。
    candidate_bitmap_inner_ready_priority_weight: float = 1.0

    # 64 条候选边的纯随机消融开关，方向与普通“启用功能”开关不同：
    # False：使用 48 条评分边 + 16 条随机边，并保留 strict/weak/try、评分和配额流程；
    # True：关闭三类别、评分和配额对采样的影响，从 strict/weak/try 全集等概率随机取 64 条边。
    # 两种模式都只硬排除 probing_bitmap=1 的并发探测边；已成功/确认失败边允许再次进入候选池。
    # PPO 开启时从当前 worker 分片采样；PPO 关闭时从全图采样，不给 base 使用分片。
    # 该开关只改变 64 条候选边的构成，不改变后续 PPO 从候选池中选择目标边的流程。
    candidate_bitmap_random_pool_enabled: bool = False

    # PPO 对“64 选 1”的控制开关，同时控制 PPO 专属的多环境目标边分片：
    # True：32 个环境先按已知结构均衡分片，各自在自己的目标边分片中生成64条候选，
    #       使用 PPO 给出的候选编号，并只对最终选中的1条边申请轻量共享预占。
    # False：不分片、不预占并忽略 PPO 编号；纯随机 base 均匀随机选边，
    #        非 base 消融在表示学习或 HNSW/熵生效时使用下方“评分引导 + 随机探索”。
    #        配合 candidate_bitmap_random_pool_enabled=True 时即为全图等概率随机 base。
    # 分片只限制目标边选择；正式图遍历、下游构造、HNSW和callback共享状态均不受限制。
    # False 只关闭 PPO 对实际选边的影响，PPO 训练外壳、环境步数、回调和统计仍照常运行。
    candidate_bitmap_ppo_selection_enabled: bool = True

    # PPO 关闭、非纯随机 base，且表示学习或 HNSW/熵实际生效时的64选1策略：
    # 0.70 表示 70% 按当前综合评分的 softmax 权重选择，剩余 30% 在有效去重候选中均匀随机。
    # 该参数不作用于 PPO 选边或 candidate_bitmap_random_pool_enabled=True 的纯随机 base。
    candidate_bitmap_guided_selection_probability: float = 0.70

    # 引导选择的 softmax 温度；越小越偏向高分边。评分会先在当前候选池内归一化到 [0, 1]。
    candidate_bitmap_guided_selection_temperature: float = 0.50

    # candidate_bitmap 模式下，每条候选边的特征维度。
    # 0: edge_score 归一化
    # 1: pattern_transfer_score
    # 2: candidate_level strict/weak/try
    # 3: outer node 是否已在正式图中
    # 4: inner node 是否已在正式图中
    # 5: 是否自环
    # 6: slot_index 归一化
    # 7: support_rank 归一化
    # 8: exploration_done：全局已成功或已确认失败，1 表示探索已经结束
    # 9: probing：该边是否正在临时 probe 图中并发探索
    # 10: local_tested：当前 env 本地是否已经测试过该边
    # 11: frontier stagnation pressure
    # 12: HNSW 相似邻居的历史成功倾向（0.5 表示没有足够历史信息）
    # 13: 结构熵（0 表示简单，1 表示复杂）
    # 14: inner_ready（目标 inner 的下游是否可完全沿正式图构造；自环候选固定为 0）
    candidate_bitmap_feature_dim: int = 15

    # candidate_bitmap 固定周期全量重载间隔。
    # 0：关闭按 step 重读 edge_index / bitmap / graph，运行态 observed、confirmed failed
    # 和新增正式图边由跨环境共享状态立即同步；表示迁移/HNSW 文件刷新后再事件式全量重载。
    # 正数：保留旧的按本地 env step 周期重载行为，仅用于调试共享状态。
    candidate_bitmap_reload_interval_steps: int = 0

    # candidate_bitmap 候选池不足时是否允许重复填充。
    candidate_bitmap_allow_pool_padding: bool = True

    # 旧版兼容参数：新版不再硬过滤本 env 已测试边，改由 local_tested 轻微降权。
    candidate_bitmap_prefer_untested: bool = True

    # env step 是否直接写回 observed/failed/probing bitmap。
    # 默认 False：避免多个并行 env 同时写 candidate_bitmaps.npz 产生竞争。
    # 后续可以由 manager 或带锁写入器统一更新。
    candidate_bitmap_update_npz_in_env: bool = False

    # env 构造 probe_graph/probe_predicate 的临时输出目录。
    # 仅 debug / dry-run 使用；正式 RL 默认不落盘 runtime probe。
    candidate_bitmap_runtime_probe_dir: Path = CANDIDATE_BITMAP_DIR / "runtime_probe"

    # 阶段17-11：正式 RL 是否持久化每一步的 probe_graph/probe_predicate 调试文件。
    # 默认 False，避免 output/candidate_bitmap/runtime_probe 按 pid/step 爆炸增长。
    candidate_bitmap_runtime_persist_probe_artifacts: bool = False

    # 若开启 persist_probe_artifacts，可设置采样间隔；0 表示不采样。
    # 例如 1000 表示每个 env 每 1000 个本地 step 最多落盘一次。
    candidate_bitmap_runtime_probe_artifact_sample_every: int = 0

    # 阶段17-12：每个 env 内缓存 edge_id -> probe predicate 构造结果。
    # 避免同一 candidate edge 重复执行 probe graph / predicate 构造。
    candidate_bitmap_probe_template_cache_size: int = 5000

    # 阶段17-12：candidate_bitmap 模式下 callback 的 snapshot / SYNC_PRINT 间隔。
    # 原先 200 太频繁，会频繁打断 Master pipeline。
    candidate_bitmap_snapshot_save_every_steps: int = 2048

    # 阶段17-13：正式训练中是否把 probe_graph_report / probe_predicate_report 放入 info。
    # 默认 False，避免 32 个 SubprocVecEnv worker 每步传输大调试对象。
    # dry-run / smoke 仍可通过 candidate_bitmap.py 输出完整 report。
    candidate_bitmap_include_probe_reports_in_info: bool = False

    # 阶段17-13：candidate_bitmap 模式下 checkpoint 保存频率。
    # 原先固定 200，对长时间训练会产生较多 zip 写盘；这里只降低 I/O，不影响训练逻辑。
    candidate_bitmap_checkpoint_save_freq: int = 10000

    # 是否启用主进程 callback 集中回写 candidate bitmap。
    candidate_bitmap_callback_update_enabled: bool = True

    # 每隔多少 SB3 timestep 批量 flush 一次 candidate bitmap 结果。
    # 从 2048 温和缩短到 1024，在批量 I/O 与及时同步候选状态之间折中。
    candidate_bitmap_callback_flush_steps: int = 1024

    # 至少累计多少条 candidate bitmap 结果才执行 flush。
    # 与 flush 间隔同步缩小，避免较小批次长期滞留在 callback 中。
    candidate_bitmap_callback_min_batch: int = 256

    # callback 成功回写 bitmap 后，是否立即通知 env worker 重新读取最新 bitmap。
    # 阶段17-11：默认关闭，worker 依赖 candidate_bitmap_reload_interval_steps 周期 reload。
    candidate_bitmap_callback_sync_workers_after_flush: bool = False

    # ================== 【阶段17-8：candidate bitmap frontier 动态开放】 ==================
    # 是否允许 candidate_bitmap 模式在 frontier 停滞时动态开放 weak / try 候选。
    # 注意：这里不修改 candidate_bitmaps.npz，只在 env 采样候选池时临时扩大可选范围。
    candidate_bitmap_frontier_open_enabled: bool = True

    # candidate_bitmap 模式至少运行多少 env step 后，才允许 frontier 动态开放。
    # 防止训练刚开始时过早引入 weak / try。
    candidate_bitmap_frontier_open_warmup_steps: int = 500

    # frontier_mode_active 且停滞压力达到该阈值后，开放 weak 候选。
    candidate_bitmap_frontier_open_weak_pressure: float = 0.50

    # frontier_mode_active 且停滞压力达到该阈值后，开放 try 候选。
    candidate_bitmap_frontier_open_try_pressure: float = 0.85

    # frontier 模式下候选池中 weak 候选的最低占比。
    # 只有 weak 已开放且存在可用 weak 候选时生效。
    candidate_bitmap_frontier_weak_min_ratio: float = 0.25

    # frontier 模式下候选池中 try 候选的最低占比。
    # 只有 try 已开放且存在可用 try 候选时生效。
    candidate_bitmap_frontier_try_min_ratio: float = 0.10

    # frontier 模式下 weak / try 的优先级加权。
    # pressure 越高，weak / try 越容易进入候选池。
    candidate_bitmap_frontier_level_priority_boost: float = 1.20

    # frontier 模式下 strict 的轻微降权。
    # 不是禁用 strict，只是防止 strict 长期占满候选池。
    candidate_bitmap_frontier_strict_pressure_penalty: float = 0.25

    # ================== Stage 21-4: exploration fairness + controlled free exploration ==================
    # 探索公平化：
    #   RL 采样候选池时，core_internal / core_to_extension / extension_outer 尽量公平进入 pool。
    #   统计阶段仍然保留 edge_group，用于区分初始 30 节点区域和新增扩展节点区域。
    # True：按下方比例执行 edge group 配额；False：不执行该分组配额。
    candidate_bitmap_group_fair_sampling_enabled: bool = True
    candidate_bitmap_group_fair_sampling_warmup_steps: int = 0
    # 最初设定为 core_internal/core_to_extension/extension_outer = 34%/33%/33%；
    # 当前进一步倾斜到未决边较多的 extension_outer，减少从未进入测试的扩展区域边。
    candidate_bitmap_group_fair_target_ratios: tuple = (
        ("core_internal", 0.15),
        ("core_to_extension", 0.15),
        ("extension_outer", 0.70),
    )

    # 受控自由探索：
    #   不改变 candidate universe，不恢复 HOLE / VALUE_SINGULAR / 类型包装器 / NOT IN；
    #   只在 runtime candidate pool 中更早开放已有 weak / try candidate bitmap。
    candidate_bitmap_fair_exploration_open_enabled: bool = True
    candidate_bitmap_fair_exploration_warmup_steps: int = 0
    candidate_bitmap_fair_exploration_open_weak: bool = True
    candidate_bitmap_fair_exploration_open_try: bool = True
    candidate_bitmap_fair_exploration_try_warmup_steps: int = 2000
    # weak/try 只约束 48 条评分候选；16 条随机候选不受该最低比例限制。
    candidate_bitmap_fair_exploration_weak_min_ratio: float = 0.30
    candidate_bitmap_fair_exploration_try_min_ratio: float = 0.05

    # 21-1 / 21-3 修复后，部分旧 failed edge 可能已经可以构造；
    # 这里只允许重试构造类失败，不重试所有 DB 语义失败。
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

    # Stage 20-5 legacy switch:
    #   关闭旧的 core -> extension 偏置，避免探索阶段 core 长期占据 60% pool。
    #   保留这些字段只是为了兼容旧代码路径；21-4 主路径使用 group_fair_sampling。
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
    # ======================================================================


    # ======================================================================


    # 候选排序时加入的小扰动，避免完全固定顺序导致 PPO 只学会总选第 0 个。
    candidate_pool_priority_jitter: float = 0.02


    # 正式训练默认不执行 check_env，避免 check_env 真实 step 造成训练前副作用。
    # 如需本地调试 action_space / observation_space，可临时改为 True。
    enable_env_compliance_check: bool = False
    # ======================================================================

    # ================== 【阶段 11：frontier-aware 覆盖反馈引导】 ==================
    # 是否启用 frontier mode。
    # frontier mode 不生成手写模板，只根据当前 observed_edges 反馈调整候选排序和 reward。
    frontier_mode_enabled: bool = True

    # 每个 env 内用于判断“新边停滞”的滑动窗口长度。
    # 注意：这是单个 env 的本地 step 数，不是全局 32 env 合计 step。
    frontier_stagnation_window_steps: int = 200

    # 窗口内至少有多少 DB 成功样本，才允许判定为 frontier 停滞。
    # 避免因为整体生成质量差而误判为“需要追新边”。
    frontier_min_db_success_in_window: int = 30

    # 窗口内至少有多少 parser 成功样本，才允许判定为 frontier 停滞。
    frontier_min_parse_success_in_window: int = 20

    # 窗口内实际新边数小于等于该值时，才认为出现新边停滞。
    frontier_max_new_edges_in_window: int = 0

    # 如果窗口内新边数达到该值，则退出 frontier mode。
    frontier_exit_new_edges_in_window: int = 2


    # ======================================================================


    # ======================================================================

    # ===============================================================


    reward_execution_error: float = -1.0  # DB 执行失败惩罚


    reward_low_freq_operator_threshold: int = 5


    # ======================================================================

    # ================== 【阶段17-7：candidate bitmap 专用 reward】 ==================
    reward_candidate_bitmap_valid_sql_bonus: float = 0.5
    reward_candidate_bitmap_parser_success_bonus: float = 0.5

    # 仅跨环境原子判定出的“全局首次成功边”获得该奖励；重复成功不会再次获得。
    reward_candidate_bitmap_global_new_edge_bonus: float = 10.0

    # parser 抽边后命中本次 target edge 的奖励。
    reward_candidate_bitmap_target_hit_bonus: float = 1.0

    # DB 成功、parser 成功，但没有命中 target edge 的惩罚。
    reward_candidate_bitmap_target_miss_penalty: float = -2.0

    # DB 已执行但目标边无效、以及下游无法构造时的不同惩罚。
    reward_candidate_bitmap_db_invalid_penalty: float = -2.0
    reward_candidate_bitmap_downstream_unavailable_penalty: float = -1.0

    # 已全局探索结束的边统一惩罚；尚未结束但当前 env 测试过的边只轻微惩罚。
    reward_candidate_bitmap_completed_edge_penalty: float = -5.0
    reward_candidate_bitmap_local_unfinished_retry_penalty: float = -0.25

    # candidate_bitmap 的语法错误或 placeholder/HOLE 统一重罚。
    reward_candidate_bitmap_syntax_placeholder_penalty: float = -5.0

    # 以下候选先验只负责构成输入和候选排序，不再直接发 reward，避免 PPO
    # 通过复述人工先验取代对“首次成功、未探索结束、下游可构造”的学习。
    reward_candidate_bitmap_edge_score_scale: float = 0.0

    # probe edge 可构造性 support_rank 辅助奖励缩放。
    reward_candidate_bitmap_support_rank_scale: float = 0.0

    # strict / weak / try 层级小奖励。
    reward_candidate_bitmap_strict_level_bonus: float = 0.0
    reward_candidate_bitmap_weak_level_bonus: float = 0.0
    reward_candidate_bitmap_try_level_bonus: float = 0.0

    # frontier mode 下，weak / try 命中 target edge 时的额外探索奖励。
    reward_candidate_bitmap_frontier_weak_try_bonus: float = 0.0

    # 旧字段保留兼容读取；candidate_bitmap 新奖励分支使用上面的
    # reward_candidate_bitmap_local_unfinished_retry_penalty。
    reward_candidate_bitmap_repeated_edge_penalty: float = -0.25

    # 旧版兼容参数：新版统一使用 candidate_bitmap_exploration_done_penalty。
    reward_candidate_bitmap_observed_repeat_penalty: float = -2.0

    # 旧版兼容参数：新版不会把尚未确认的本地失败直接视为探索结束。
    reward_candidate_bitmap_failed_before_penalty: float = -1.5
    # ======================================================================

    # ================== 【覆盖率逼近模式】 ==================
    coverage_mode: str = "fixed_base"
    # =========================================================
    # ==================================================================

    # ================== 【RuntimeStats 并发写入控制】 ==================
    # 是否允许每个 env worker 在 step 中直接写运行时统计文件。
    #
    # 默认 False：
    #   32 个并行环境如果每步都直接写 structure_attempts.json，
    #   会高频竞争同一个 .lock 文件，导致大量 lock timeout。
    #
    # 正常训练时：
    #   Correct 成功谓词会通过 shared_queue -> pkg_manager 统一记录；
    #   env 侧失败/placeholder 尝试默认不落盘，只放入 info。
    #
    # 单环境调试时如果确实想记录所有失败尝试，可以临时改为 True。
    record_env_step_runtime_stats: bool = False

    # Stage-2 joint type-signature evidence is aggregated in memory. It does
    # not alter actions, rewards, graph insertion, or the training schedule.
    stage2_joint_signature_evidence_enabled: bool = True
    # Flush after this many parent/operand observations, rather than per SQL.
    stage2_joint_signature_flush_events: int = 5000
    # Bound the compact evidence index so long experiments cannot grow it
    # without limit. High-frequency combinations are retained first.
    stage2_joint_signature_max_entries: int = 50000
    # =================================================================


    # 初始正确谓词文本路径。
    init_correct_predicates_path: Optional[Path] = field(
        default_factory=lambda: get_database_data_path("Correct_Collection.txt")
    )


    # 共享算子覆盖配置。
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


    # ==================================================================


# ====================== 4. Actor-Critic 算法配置（Stable-Baselines3） ======================

@dataclass
class RLAlgoConfig: # Stable-Baselines3 PPO 的训练参数。

    # 训练器固定使用 PPO；该字段仅用于显示算法名称。
    algo_name: Literal["PPO"] = "PPO"

    # 学习率
    learning_rate: float = 3e-4

    # 折扣因子 gamma
    gamma: float = 0.99

    # 每次 rollout 的步数
    n_steps: int = 128

    # PPO 更新参数
    batch_size: int = 64
    n_epochs: int = 4
    clip_range: float = 0.2

    # 并行环境数量。
    #
    # 每个 env 都会：
    #   1. 维护候选池；
    #   2. 连接 MySQL；
    #   3. 执行 SQL；
    #   4. 通过 shared_queue 把成功谓词交给后台图谱进程。
    #
    # 当前保留 32 个并行环境：
    #   1. 原项目已经多次使用 32 环境稳定运行；
    #   2. candidate_bitmap 模式每步构造候选边谓词，单步更轻；
    #   3. 保持 32 并发可以减少墙钟时间。
    #
    # 如果后续发现 MySQL、shared_queue 或 PKG 图谱进程压力过大，
    # 再考虑降到 16。
    n_envs: int = 32

    # 总训练步数（可在 train_actor_critic.py 里覆盖）
    total_timesteps: int = 400000

    # SB3 PPO 的策略网络设备。
    # auto 会沿用 Stable-Baselines3 的自动设备选择；
    # 这里仅影响 PPO actor/critic，不影响第一阶段 Transformer 使用 GPU。
    sb3_device: str = "auto"

    # 随机种子（为了可复现）
    seed: int = 42

    # 是否启用消融实验进度统计。关闭后不生成进度 CSV，也不改变原有训练流程。
    progress_log_enabled: bool = True

    # 每累计多少个全局环境步记录一次。SB3 会累计所有并行环境的步数；
    # 当前 32 个环境各执行 1 步时，全局步数增加 32。
    progress_log_interval_steps: int = 2048

    # 由 STAGE2_MODE 和 DB_CONFIG.db_type 自动生成，无需手动修改。
    experiment_name: str = ""

    # 每次运行的独立进度 CSV 保存目录。
    progress_log_dir: Path = RL_DIR / "rl_logs" / "ablation_progress"

RL_ALGO_CONFIG = RLAlgoConfig()





# ====================== 6. DBMS 配置 ======================

@dataclass
class DBConfig: # 配置结构；实际连接参数请填写 Robusta/configuration.json。
    # "mysql" / "postgresql" / "tidb" 等，仅用于标识
    db_type: str = "postgresql"

    # 关闭 Socket，改用本地 TCP 连接
    unix_socket: Optional[str] = None

    # 主机和端口
    host: str = ""
    port: int = 0

    # 用户名和密码
    user: str = ""
    password: str = ""

    # 实际的数据库名
    database: str = ""

    # 连接超时时间（秒）
    connect_timeout: int = 5

    # 单条 SQL 执行的超时时间（秒）
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
    # Field descriptions and the Stage-3 step are not database connection options.
    values.pop("_comment_db_type", None)
    values.pop("_comment_database", None)
    values.pop("_comment_step", None)
    values.pop("step", None)
    expected_fields = {item.name for item in fields(DBConfig)}
    missing = expected_fields - values.keys()
    unknown = values.keys() - expected_fields
    if missing or unknown:
        raise ValueError(
            f"Invalid fields in {config_path}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for name in ("db_type", "host", "user", "password", "database"):
        if not isinstance(values[name], str):
            raise ValueError(f"{config_path}: {name} must be a JSON string.")
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

if __package__:
    from .stage2_mode import apply_stage2_mode as _apply_stage2_mode
else:
    from stage2_mode import apply_stage2_mode as _apply_stage2_mode

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


# ====================== 6.5 Stage 3 / PTSC 配置 ======================

@dataclass
class Stage3Config:
    """第三阶段固定深度生成与 WHERE-TLP 验证配置。"""

    # WHERE-TLP only: choose compatible same-type columns per occurrence.
    # False restores one column per type marker; exploration keeps that binding.
    stage3_tlp_independent_columns_enabled: bool = True

    # Fixed-depth timed enumeration; otherwise retain ordinary stepwise sampling.
    stage3_timed_enumeration_enabled: bool = False
    stage3_timed_enumeration_depth: int = 3

    # The enumeration generator stops at this phase limit and commits its
    # partial output. TLP generation and database validation may use the
    # remaining time up to stage3_timed_enumeration_total_limit_seconds.
    stage3_timed_enumeration_generation_limit_seconds: float = 1800.0
    stage3_timed_enumeration_total_limit_seconds: float = 10800.0

    # Dedicated Stage-3 localized-random experiment switch. True selects this
    # branch; False leaves the normal and enumeration Stage-3 paths unchanged.
    # This is unrelated to every Stage-2 exploration/enumeration switch.
    stage3_timed_random_enabled: bool = False
    stage3_timed_random_depth: int = 3

    # Localized-random generation stops after half an hour. TLP generation and
    # database validation use the remaining time in the three-hour Stage-3 budget.
    stage3_timed_random_generation_limit_seconds: float = 1800.0
    stage3_timed_random_total_limit_seconds: float = 10800.0

    # 仅第三阶段随机实验：每层有效非叶布尔根按组合数升序，按数量等分小/中/大三档。
    # 总配额6表示每档随机选2个；整层不足6个时全选。固定种子保证相同图的选根可复现。
    stage3_random_seed: int = 42
    stage3_random_root_pool_size: int = 6
    # 旧候选上限仅保留兼容/报告，不再截取前16个；三档覆盖本层全部有效根节点。
    stage3_random_root_candidate_pool_size: int = 16

    # Each selected root samples with replacement inside one contiguous local
    # combination window. Repeated predicates are intentionally allowed.
    # A zero ratio keeps each root on a fixed 250,000-combination window
    # instead of expanding the window with the root's huge total count.
    stage3_random_local_window_ratio: float = 0.0
    stage3_random_local_window_min_size: int = 250000

    # Pick a root once per block instead of once per predicate. This bounds
    # signature-span metadata and keeps long-run bookkeeping/I/O inexpensive.
    stage3_random_root_block_size: int = 32768

    # Prefer configuration.json; missing or blank step defaults to 64.
    stage3_sampling_step: float = _load_stage3_sampling_step()

    # 精确组合计数对象的估算内存安全预算，单位 MB；小于等于 0 表示关闭保护。
    # 只在尚未产生本轮谓词前防止超深 DAG 的 Python 大整数耗尽内存，
    # 不截断已开始的固定步长采样，也不减少任何已经生成的输出文件内容。
    stage3_count_memory_budget_mb: int = 256

    # 第三阶段生成谓词的控制台进度间隔；小于等于 0 表示关闭进度行。
    # 只控制输出频率，不改变实际采样数量。
    stage3_predicate_progress_interval: int = 1000

    # 即使短时间内跨过多个进度间隔，控制台进度最多每隔该秒数输出一次。
    # 只抑制长期运行时的日志爆炸，不改变谓词文件、采样步长或生成数量。
    stage3_predicate_progress_min_seconds: float = 5.0

    # 第三阶段中间文本/报告文件的缓冲区大小，单位为字节。
    # 只减少频繁小块 I/O，不改变文件内容。
    stage3_output_buffer_bytes: int = 1048576

    # 是否将第三阶段采样谓词以 .txt.gz 无损保存。
    # True 只改变存储格式；解压后的谓词文本、顺序、数量和固定步长采样结果均不变。
    stage3_compress_predicate_output: bool = True

    # 采样谓词 gzip 压缩级别 1-9；默认级别 1 优先保证长期生成速度。
    # 该参数不限制输出数量，也不改变任何谓词内容。
    stage3_predicate_compression_level: int = 1

    # sampler 接收到截止时间和下游成本模型时，为 TLP 生成和数据库扫描动态预留时间。
    # 该开关只调整末轮停止采样的时机，不设置每轮采样上限、不改变采样密度或数据库判定。
    stage3_downstream_reserve_enabled: bool = True

    # 尚无完整轮次实测成本时使用的最小预留时间，单位为秒。
    stage3_downstream_reserve_min_seconds: float = 120.0

    # 对最近完整轮次测得的 TLP 生成与数据库扫描成本应用的安全系数。
    stage3_downstream_reserve_safety_factor: float = 1.25

    # 为文件关闭、状态写入和固定初始化成本额外预留的秒数。
    stage3_downstream_reserve_fixed_seconds: float = 30.0

    # 每生成多少条谓词重新检查一次预留时间；降低检查频率可避免热循环额外开销。
    # 该参数只影响检查粒度，不改变固定步长和已命中的谓词顺序。
    stage3_downstream_reserve_check_interval: int = 1000

    # 是否将第三阶段完整 TLP 测试用例以 .sql.gz 流式保存。
    # True 只改变无损存储格式：解压后仍是原有完整 SQL，谓词、顺序和数据库测试逻辑不变；
    # False 保持旧的未压缩 .sql 输出，扫描器同时兼容两种格式。
    stage3_compress_test_cases: bool = True

    # gzip 压缩级别 1-9；1 优先生成/扫描速度，适合长期第三阶段运行。
    # 该参数只在 stage3_compress_test_cases=True 时生效，不需要安装额外依赖。
    stage3_test_case_compression_level: int = 1

    # 第三阶段报告轮转保留的最近轮次数；小于等于 0 表示不自动清理旧报告。
    # 只作用于 stage3 生成的报告文件，不影响知识图、谓词或数据库数据。
    stage3_report_retention_count: int = 20

    # Stage-3 knowledge-graph sampling controls.
    # Similar successors are placed next to each other before stepwise sampling.
    stage3_similarity_order_enabled: bool = True
    # Prefer edge combinations seen together in successful predicates.
    stage3_success_guided_sampling_enabled: bool = True
    # Build this many candidates for every one predicate kept in the final output.
    stage3_candidate_pool_multiplier: int = 2
    # Fraction selected by success score; the remainder is sampled randomly.
    stage3_guided_candidate_ratio: float = 0.70

    # Stage-2 joint-signature evidence only changes the per-root sampling density.
    # Every root remains reachable and each root still uses one fixed step; the
    # bounded factor prevents sparse evidence from dominating exploration.
    stage3_evidence_weight_strength: float = 0.25
    stage3_evidence_min_observations: int = 4

    # PostgreSQL cannot resolve expressions such as NULL / NULL without context.
    # When enabled, Stage 3 casts only NULL leaves whose parent signature supplies
    # an unambiguous concrete type; ordinary literals and implicit casts are untouched.
    stage3_contextual_null_cast_enabled: bool = True

    # Reuse the DBMS rule file's existing safe argument literals only when the
    # sampled child is already a literal leaf. The exploration ratio preserves
    # unrestricted boundary values and never replaces nested predicate branches.
    stage3_safe_value_guidance_enabled: bool = True
    stage3_safe_value_exploration_ratio: float = 0.01

    # Build an in-memory type-aware projection after loading the Stage-2 graph
    # and before DAG unfolding. The source graph file is never rewritten.
    stage3_type_projection_enabled: bool = True
    # PostgreSQL uses live catalog signatures plus Stage-2 joint evidence.
    # Other DBMSs remain pass-through until a catalog provider is registered.
    stage3_type_projection_catalog_enabled: bool = True
    stage3_type_projection_evidence_enabled: bool = True
    # One successful observation is admissible because DBMS behavior may
    # expose coercions that are not represented by catalog casts.
    stage3_type_projection_min_success_count: int = 1

    # A combination is treated as known-invalid only after repeated Stage-2
    # type failures and no success. This is evidence classification only;
    # catalog-compatible and empirically successful signatures stay enabled.
    stage3_type_projection_known_invalid_min_observations: int = 4

    # Catalog-unconfirmed graph paths use a separate low-density sampling
    # channel. This preserves discovery of undocumented DBMS coercions without
    # letting unknown combinations dominate normal exact-signature sampling.
    stage3_type_projection_unknown_exploration_ratio: float = 0.01
    # Keep the unknown exploration channel bounded for every DBMS. A value of
    # one (the default) creates at most one disjoint rectangle per operator;
    # two is allowed for a slightly broader experiment. This prevents a large
    # Stage-2 evidence set from fragmenting the projection graph into thousands
    # of variants. The selected rectangle contains no recorded tuple.
    stage3_type_projection_max_unknown_variants: int = 1

    # 仅 MySQL 第三阶段生效：每个原始算子最多保留的投影变体数。
    # 用于限制图规模，不改变 PostgreSQL、第一阶段或第二阶段。
    stage3_mysql_max_projection_variants: int = 96

    # 仅 Oracle 第三阶段生效：有界扩大每个原始算子的投影变体数。
    # Oracle 使用自己的证据权重和保守合并白名单，不改变其他数据库分支。
    stage3_oracle_max_projection_variants: int = 192

    # MySQL evidence is intentionally gentle: implicit conversions remain
    # reachable, while weak failures only reduce sampling density slightly.
    # The values are fractional failure weights; unknown combinations also
    # retain 90% density so they remain a meaningful implicit-cast exploration
    # channel. Extend the DBMS classifier in RL/joint_signature_evidence.py when
    # adding a DBMS.
    stage3_mysql_evidence_weight_strength: float = 0.12
    stage3_mysql_value_failure_weight: float = 0.05
    stage3_mysql_execution_failure_weight: float = 0.02
    stage3_mysql_unknown_evidence_factor: float = 0.90

    # Oracle-only Stage-3 evidence controls. Oracle rejects incompatible
    # operand types more strictly than MySQL, so its type failures remain
    # strong evidence and value/execution failures use an independent scale.
    # The implicit-conversion ratio is intentionally separate from MySQL's
    # relaxed unknown-signature channel.
    stage3_oracle_safe_value_exploration_ratio: float = 0.05
    stage3_oracle_implicit_conversion_exploration_ratio: float = 0.05
    stage3_oracle_evidence_weight_strength: float = 0.25
    stage3_oracle_type_failure_weight: float = 1.00
    stage3_oracle_value_failure_weight: float = 0.40
    stage3_oracle_execution_failure_weight: float = 0.20

    # SQL Server uses the same bounded unknown-signature channel as Oracle,
    # with its own control so the other DBMS sampling settings stay unchanged.
    stage3_sqlserver_implicit_conversion_exploration_ratio: float = 0.05

    # 按数据库真实列类型拆分 Stage-3 的 COL_* 终端，并传递给 SQL 实例化。
    # 这里不会增加显式转换；Oracle 使用精确目录签名和 Stage-2 运行证据。
    stage3_exact_column_type_guidance_enabled: bool = True
    # Exact hints normally select the same live source type. This small random
    # channel preserves opportunities to discover undocumented DBMS coercions.
    stage3_column_type_exploration_ratio: float = 0.01

    # Stage-3 database results reweight the same root type signature in a later
    # run. The tier boundaries target executable predicates rather than merely
    # rewarding anything above 50%: highly reliable signatures retain their
    # density, while repeatedly weak signatures receive progressively fewer
    # samples. The 0.01 floor is a near-block, not a graph deletion, so a tiny
    # implicit-conversion exploration channel remains available.
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

    # Legacy linear feedback remains available when tiered feedback is
    # explicitly disabled, which keeps older experiment configurations usable.
    stage3_runtime_signature_feedback_strength: float = 0.90
    stage3_runtime_signature_feedback_min_factor: float = 0.10

    # MySQL-only Stage-3 runtime feedback weights. Type/collation failures
    # (3854/3995) are strong negative evidence; value/regex failures and
    # unclassified execution errors stay weak so implicit conversions remain
    # reachable. PostgreSQL and other DBMS paths use the generic feedback.
    stage3_mysql_runtime_type_failure_weight: float = 1.00
    stage3_mysql_runtime_value_failure_weight: float = 0.15
    stage3_mysql_runtime_other_failure_weight: float = 0.05
    stage3_mysql_runtime_timeout_weight: float = 0.10

    # Oracle-only runtime feedback weights. ORA type/value/execution errors
    # and timeout observations are kept separate from MySQL's loose-cast
    # policy; execution errors are represented by other_failure_count.
    stage3_oracle_runtime_type_failure_weight: float = 1.00
    stage3_oracle_runtime_value_failure_weight: float = 0.40
    stage3_oracle_runtime_other_failure_weight: float = 0.20
    stage3_oracle_runtime_timeout_weight: float = 0.05

    # Stage-3-only component feedback. A bounded block merges the operator
    # signatures seen on a few recursive paths; it avoids storing one trace
    # per predicate while allowing deep-node failures to affect later rounds.
    stage3_component_signature_feedback_enabled: bool = True
    stage3_component_feedback_trace_stride: int = 16
    stage3_component_feedback_block_size: int = 256
    stage3_component_feedback_max_keys_per_block: int = 32

    # Write one compact time-series row at this wall-clock interval. Database
    # workers only report through the existing 1000-case progress boundary, so
    # this does not add per-SQL disk I/O.
    stage3_metrics_enabled: bool = True
    stage3_metrics_interval_seconds: float = 600.0

    # TLP SQL error 样例最大写入数量。
    # 注意：这里只限制错误样例落盘数量，不影响 SQL error 总数统计。
    # 目的是避免大量不合法复杂谓词导致日志/磁盘爆炸。
    tlp_error_sample_limit: int = 300

    # 单条 SQL 在错误样例文件中的最大保留长度。
    tlp_error_sql_max_chars: int = 1000

    # 单条数据库错误信息在错误样例文件中的最大保留长度。
    tlp_error_message_max_chars: int = 500

    # 逻辑 Bug 详细报告的最大写入数量；总 Bug 数仍完整统计。
    # 达到上限后保留计数，但不再为每个 Bug 写完整 SQL 现场，避免长期运行报告爆炸。
    tlp_bug_sample_limit: int = 100

    # 单条 Bug 详细报告中 SQL 的最大保留长度。
    tlp_bug_sql_max_chars: int = 2000

    # TLP 扫描进度打印间隔。
    # 只影响控制台进度输出，不逐条写 scan_log。
    tlp_progress_log_interval: int = 1000

    # TLP 单条 SQL 的执行超时，单位秒；小于等于 0 表示关闭。
    # 该值仅由第三阶段 TLP 执行器显式启用，不改变前两阶段的数据库行为。
    tlp_query_timeout: int = 10

    # 第三阶段 TLP 数据库验证前，对可能放大字符串的函数长度参数做执行时限幅。
    # 原始谓词和 TLP 文件保持不变；该开关只改变实际送入数据库的危险参数值。
    tlp_resource_argument_bounding_enabled: bool = True
    # REPEAT(text, count) 的重复次数限制为 0..该值。
    tlp_repeat_count_max: int = 1024
    # REPEAT 的实际次数上限还会结合输入长度计算；输入已达到该字符预算时次数最多为 1，
    # 因而深层 REPEAT(REPEAT(...)) 不会在每层继续指数放大。
    tlp_repeat_output_max_chars: int = 4096
    # LPAD/RPAD/SUBSTR 的长度限制为 0..该值；LEFT/RIGHT 保留正负语义并限制绝对值。
    tlp_string_length_max: int = 4096
    # 每轮最多在现有 scan_log 中保留多少条“原 SQL -> 实际执行 SQL”样例。
    # 总改写数始终统计；限制样例数可避免长期运行时产生额外 I/O 和输出爆炸。
    tlp_resource_bound_sample_limit: int = 5

    # Stage-3 MySQL-only retry for error 1690. The original predicate/TLP files
    # stay unchanged; only EXP arguments in the database execution copy are
    # bounded before one retry. PostgreSQL and other DBMS paths ignore this.
    tlp_mysql_overflow_retry_enabled: bool = True
    tlp_mysql_exp_argument_limit: float = 700.0

    # TLP 多进程任务分块大小。较小的分块可减少慢 SQL 对同组任务的阻塞，
    # 但不会改变进度日志频率或增加磁盘写入。
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





# ====================== 8. Master-Worker & 监控配置 ======================
@dataclass
class MonitorConfig:
    # 监控日志路径（记录每次发现新边的时间、步数、边数）
    monitor_log_path: Path = OUTPUT_DIR / "monitor_log.csv"
    """
    # 最终全局正确谓词的持久化文本路径（代替原来的 init_correct_predicates_path 直接追加写）
    global_correct_txt_path: Path = OUTPUT_DIR / "Global_Correct_Collection.txt"
    """
    # 【核心修改】：直接复用原先的正确集合路径，保证强化学习读取和写入位置绝对统一！
    global_correct_txt_path: Path = RL_ENV_CONFIG.init_correct_predicates_path

    # 全局大图的序列化保存路径（供第三阶段直接读取，解耦阶段二和三）
    global_pkg_pkl_path: Path = OUTPUT_DIR / "global_pkg.pkl"

    # I/O 刷盘的 batch size，保护 HPC 共享存储
    io_batch_size: int = 300

    # ================== 【PKG 图谱解析并行配置】 ==================
    # global_graph_manager 中用于并行解析 Correct 谓词的 worker 数量。
    #
    # 注意：
    #   这里只并行 analyzer.analyze()；
    #   builder.add_predicate()、runtime_stats 写入、global_knowledge_graph.json 保存
    #   仍然由主图谱进程串行完成，避免多进程同时改图或抢写 JSON。
    #
    # 建议：
    #   工作站 CPU 核心较多时可以设为 4 或 8；
    #   如果发现和 MySQL / RL env 抢 CPU，可降为 2；
    #   如果要完全关闭并行解析，可设为 1。
    pkg_parse_workers: int = 4

    # batch 小于该值时使用串行解析，避免多进程调度开销大于收益。
    pkg_parse_min_batch: int = 64
    # ============================================================


MONITOR_CONFIG = MonitorConfig()

# ====================== 9. Predicate Recycle 回炉配置 ======================
@dataclass
class RecycleConfig:
    """
    阶段 6C：RL 训练结束后的谓词回炉配置。

    注意：
        这里不负责第一阶段重训；
        这里只控制是否在 RL 结束后调用 RL/predicate_recycle.py
        生成 recycle_candidates.json / recycle_apply_log.json。
    """

    # 是否在一次 RL 训练结束后自动运行回炉流程。
    enable_after_training: bool = True

    # ================== 【阶段 6C 补丁：周期性回炉触发】 ==================
    # 是否启用周期性触发器。
    # 启用后，RL 不必跑完整个 total_timesteps，而是在 recycle_interval_steps
    # 到达后优雅停止当前 model.learn()，然后进入 finally 的回炉流程。
    recycle_enable_periodic_trigger: bool = True

    # 周期性触发步数。
    # 当前使用 32 个并行环境，因此 80000 / 32 = 2500，
    # 对每个环境来说大约推进 2500 步后触发一个 RL cycle 边界。
    #
    # total_timesteps=320000 且 recycle_interval_steps=80000 时，
    # auto_cycle 默认运行 4 轮：
    #   rl_cycle -> recycle apply -> recycle_pre_rl
    recycle_interval_steps: int = 80000

    # 到达周期步数后是否停止当前训练 cycle。
    # 默认 True：callback 返回 False，让 SB3 正常结束 learn()，进入 finally 保存模型。
    recycle_stop_training_after_trigger: bool = True

    # 周期性触发后，是否仍然在 finally 中执行回炉。
    # 默认 True：停止当前 RL cycle 后，关闭后台进程并运行 predicate_recycle。
    recycle_run_after_periodic_stop: bool = True
    # ====================================================================

    # 6A：候选筛选数量。
    recycle_top_k: int = 1000

    # 6A：候选最低分数。
    recycle_min_score: float = 1.0

    # 6A：是否允许纯 pkg_initial 初始谓词进入候选。
    recycle_include_initial: bool = False

    # 6A：是否要求候选具有价值信号。
    # 注意：在 predicate_recycle.py 中，这里的“贡献”已经扩展为：
    # 新边 / 新算子 / 低频结构 / 多样结构 / 稳定成功结构。
    recycle_require_new_contribution: bool = True

    # 6B：是否在训练结束后真实追加候选到 predicate.txt。
    # 默认 False，避免训练结束后自动修改第一阶段输入文件。
    recycle_apply_after_training: bool = False

    # 6B：最多追加多少条候选。
    recycle_apply_top_k: int = 500

    # 6B：训练结束后是否执行 dry-run apply。
    # 默认 True：会生成 recycle_apply_log.json，但不会修改 predicate.txt。
    # 如果希望真实写入，需要设置：
    #   recycle_apply_after_training=True
    #   recycle_dry_run_after_training=False
    recycle_dry_run_after_training: bool = True

    # 6B：真实写入 predicate.txt 前是否备份。
    recycle_create_backup: bool = True

    recycle_abstract_cap: int = 5
    recycle_apply_require_value_signal: bool = True
    recycle_disallow_high_freq_zero_signal: bool = True

    # ================== 【阶段 14：回流策略修复】 ==================
    # fixed_base 模式下，frontier_new_edges>0 也视为强贡献。
    # 目的：避免 Frontier 发现的新边因为不属于 fixed_base_structure 而无法回流。
    recycle_frontier_new_edges_as_strong: bool = True

    # fixed_base 模式下，允许 frontier_new_edges>0 的非 fixed_base 谓词进入 apply。
    # 注意：只放行真正有 frontier 新边贡献的谓词，不放行普通无新边非 fixed_base 谓词。
    recycle_allow_frontier_non_fixed_base_apply: bool = True

    # low_freq_structure 是否直接占用 value 通道。
    # 默认 False：无新边 low_freq 只进入 novel_structure 补充通道，避免挤占强贡献谓词名额。
    recycle_low_freq_structure_in_value_channel: bool = False

    # low_freq_structure 是否允许进入 novel_structure 通道。
    # 默认 True：保留少量“无新边但结构可能有价值”的谓词。
    recycle_low_freq_structure_in_novel_channel: bool = True

    # fixed_base_new_edges 的回流打分权重。
    recycle_fixed_base_new_edges_score_weight: float = 5.0

    # frontier_new_edges 的回流打分权重。
    # 与 fixed_base_new_edges 同级，表示 Frontier 真实扩图也应被优先回流。
    recycle_frontier_new_edges_score_weight: float = 5.0
    # ============================================================

    # ================== 【novel structure 回炉控制】 ==================
    recycle_enable_novel_structure_fill: bool = True
    recycle_novel_structure_target_count: int = 120
    recycle_fixed_base_require_base_signal: bool = True
    # ============================================================
    recycle_strong_target_quota: int = 250
    recycle_sparse_target_quota: int = 150
    recycle_diverse_target_quota: int = 75
    recycle_stable_target_quota: int = 25


RECYCLE_CONFIG = RecycleConfig()
# ====================================================================

# ====================== 10. Auto Cycle 自动循环配置 ======================
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

    # 是否自动根据 RL_ALGO_CONFIG.total_timesteps 和
    # RECYCLE_CONFIG.recycle_interval_steps 计算循环轮数。
    #
    # 默认 True：
    #   例如 total_timesteps=320000，recycle_interval_steps=80000，
    #   则 auto_cycle 自动运行 4 轮。
    #
    # 如果设置为 False，则使用 auto_cycle_num_cycles。
    auto_cycle_auto_compute_num_cycles: bool = True

    # 自动循环轮数。
    # 仅当 auto_cycle_auto_compute_num_cycles=False 且命令行未显式传入 num_cycles 时使用。
    #
    # 例如 3 表示执行 3 轮：
    #   rl_cycle -> recycle apply -> recycle_pre_rl
    auto_cycle_num_cycles: int = 3

    # 如果 output/predicate.txt 或 encoder_state.pt 不存在，是否先自动运行一次 pre_rl。
    # 第一次从干净工程启动时建议 True。
    auto_cycle_run_pre_rl_if_missing: bool = True

    # recycle 候选筛选数量，对应 predicate_recycle.py 的 --top-k。
    auto_cycle_recycle_top_k: int = 1000

    # recycle 候选最低分数，对应 predicate_recycle.py 的 --min-score。
    auto_cycle_recycle_min_score: float = 1.0

    # recycle 实际追加数量，对应 predicate_recycle.py 的 --apply-top-k。
    # 当 auto_cycle_recycle_dynamic_schedule_enabled=False 时使用。
    auto_cycle_recycle_apply_top_k: int = 500

    # ================== 【阶段 15：动态回流规模调度】 ==================
    # 是否在 auto_cycle 中按 cycle 轮次动态调整回流规模。
    # True:
    #   cycle 1 使用 700 value + 240 novel；
    #   cycle 2 使用 600 value + 200 novel；
    #   cycle 3 使用 500 value + 160 novel；
    #   cycle >=4 使用默认 400 value + 120 novel。
    auto_cycle_recycle_dynamic_schedule_enabled: bool = True

    # 前几轮的动态回流规模，格式为：
    #   (value_apply_limit, novel_structure_target_count)
    auto_cycle_recycle_value_novel_schedule: tuple = (
        (700, 240),
        (600, 200),
        (500, 160),
    )

    # 超出 schedule 后的默认回流规模。
    auto_cycle_recycle_default_value_after_schedule: int = 400
    auto_cycle_recycle_default_novel_after_schedule: int = 120
    # ==================================================================

    # auto_cycle 中是否允许纯 pkg_initial 初始谓词进入候选。
    auto_cycle_recycle_include_initial: bool = False

    # auto_cycle 中是否允许无新贡献候选进入筛选。
    # 默认 False，仍要求候选至少有新边 / 新算子 / 低频结构 / 多样结构 / 稳定成功结构等价值信号。
    auto_cycle_recycle_allow_no_new_contribution: bool = False

    # auto_cycle 中 recycle apply 是否只 dry-run。
    # 默认 False：真正追加到 output/predicate.txt。
    # 如果只是想测试流程而不改 predicate.txt，可以改为 True。
    auto_cycle_recycle_dry_run: bool = False

    # auto_cycle 中 recycle apply 前是否禁用备份。
    # 默认 False：保留备份。
    auto_cycle_recycle_no_backup: bool = False

    # 如果本轮 recycle apply 没有新增任何谓词，是否跳过 recycle_pre_rl。
    # 默认 True：避免无意义地重训第一阶段。
    auto_cycle_skip_retrain_if_no_apply: bool = True

    # 是否在每轮 apply 后运行 recycle_pre_rl。
    # 默认 True：实现“RL -> 回炉 -> 第一阶段重训”的闭环。
    auto_cycle_run_recycle_pre_rl: bool = True

    # 子流程失败时是否立刻停止整个 auto_cycle。
    # 默认 True：避免在前一步失败的情况下继续污染后续产物。
    auto_cycle_stop_on_failure: bool = True

    # auto_cycle 全部循环结束后，是否继续运行第三阶段。
    #
    # True：
    #   python main.py 默认流程变成：
    #       initial pre_rl if missing
    #       -> 多轮 auto_cycle
    #       -> pkg
    #       -> BugFinding
    #
    # False：
    #   只运行到第二阶段闭环结束，适合只想观察 RL/回流效果的实验。
    auto_cycle_run_stage3_after_cycles: bool = True

    # recycle_pre_rl 结束后，是否检查 operator similarity 相关输出。
    # 目的是避免第一阶段重训成功返回，但 operator_similarity.json 等文件缺失，
    # 导致下一轮 RL 仍然使用旧表或直接失败。
    auto_cycle_verify_operator_similarity_after_retrain: bool = True


AUTO_CYCLE_CONFIG = AutoCycleConfig()
# ====================================================================

