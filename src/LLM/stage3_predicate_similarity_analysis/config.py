from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SimilarityAnalysisConfig:
    ast_index_workers: Optional[int] = None
    ast_index_batch_size: int = 256

    enabled: bool = True
    fail_main_on_error: bool = False

    shingle_size: int = 3
    normalize_unquoted_tokens: bool = True

    max_predicates_to_read: Optional[int] = None
    executable_only: bool = False

    minhash_size: int = 32
    lsh_rows_per_band: int = 4

    window_radius: Optional[int] = 10
    window_target_pair_budget: int = 5_000_000
    window_radius_min: int = 2
    window_radius_max: int = 64

    lsh_bucket_fetch_multiplier: int = 4

    sqlite_insert_batch_predicates: int = 2_000
    progress_interval_predicates: int = 10_000

    detail_gzip_compression_level: int = 1
    examples_per_bucket: int = 10

    keep_work_directory: bool = False


CONFIG = SimilarityAnalysisConfig()
