from typing import Literal, Protocol


class IUserProfileGenerator(Protocol):
    async def generate(
        self,
        gender: Literal["male", "female"],
        annual_household_income_k: int,
        count: int,
    ) -> list[str]: ...
