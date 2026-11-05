from dataclasses import dataclass
from itertools import combinations


JOIN_PARAMETERS = ("enable_hashjoin", "enable_mergejoin", "enable_nestloop")
SCAN_PARAMETERS = ("enable_seqscan", "enable_indexscan", "enable_indexonlyscan")
PARAMETERS = JOIN_PARAMETERS + SCAN_PARAMETERS


@dataclass(frozen=True)
class HintConfig:

    hint_id: int
    disabled_joins: tuple[str, ...]
    disabled_scans: tuple[str, ...]

    @property
    def settings(self) -> dict[str, str]:
        if self.hint_id == 1:
            return {}
        disabled = set(self.disabled_joins + self.disabled_scans)
        return {name: "off" if name in disabled else "on" for name in PARAMETERS}


def generate_configs() -> list[HintConfig]:
    join_subsets = [
        subset for count in range(3) for subset in combinations(JOIN_PARAMETERS, count)
    ]
    scan_subsets = [
        subset for count in range(3) for subset in combinations(SCAN_PARAMETERS, count)
    ]
    return [
        HintConfig(hint_id, joins, scans)
        for hint_id, (joins, scans) in enumerate(
            ((joins, scans) for joins in join_subsets for scans in scan_subsets),
            start=1,
        )
    ]
