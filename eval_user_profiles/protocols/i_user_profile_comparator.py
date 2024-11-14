from typing import Protocol

from eval_user_profiles.models.comparison_result import ComparisonResult


class IUserProfileComparator(Protocol):
    async def compare(self, base: str, profile: str) -> ComparisonResult | None: ...
