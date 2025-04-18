from dataclasses import dataclass, field
from enum import Enum, auto

from cheesesnake.models.dataset import Dataset


class MatchType(Enum):
    TITLE = auto()
    NOTES = auto()
    ORGANIZATION = auto()
    CATEGORY = auto()


@dataclass
class SearchResult:
    dataset: Dataset
    match_types: dict[MatchType, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return sum(self.match_types.values())

    def add_match(self, match_type: MatchType, score: float) -> None:
        self.match_types[match_type] = max(score, self.match_types.get(match_type, 0))
