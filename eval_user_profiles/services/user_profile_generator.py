import asyncio
from cmd import PROMPT
from dataclasses import dataclass
from typing import Literal

from eval_user_profiles.protocols.i_azure_openai_service import IAzureOpenAIService
from eval_user_profiles.protocols.i_user_profile_generator import IUserProfileGenerator

PROMPT = """
You are a prompt engineer. You are asked to create a user profile for product testing with the following requirements:

1. Age is between {min_age} and {max_age} years old.
1. Gender is {gender}
2. This person lives in the California State of the US,
3. This person is married and has an annual household income of {annual_household_income_k} thousand dollars.
4. This person is love outdoor activities and traveling.


Please include the person name, age, demographic, personality traits.
"""  # noqa E501


@dataclass
class UserProfileGenerator(IUserProfileGenerator):
    openai_service: IAzureOpenAIService

    async def _generate(
        self,
        min_age: int,
        max_age: int,
        gender: Literal["male", "female"],
        annual_household_income_k: int,
    ) -> str:
        result = await self.openai_service.generate(
            [
                {
                    "role": "system",
                    "content": PROMPT.replace("{gender}", gender)
                    .replace(
                        "{annual_household_income_k}", str(annual_household_income_k)
                    )
                    .replace("{min_age}", str(min_age))
                    .replace("{max_age}", str(max_age)),
                }
            ],
            temperature=0.5,
        )
        return (
            result.choices[0].message.content
            if result.choices and result.choices[0].message.content
            else ""
        )

    async def generate(
        self,
        min_age: int,
        max_age: int,
        gender: Literal["male", "female"],
        annual_household_income_k: int,
        count: int,
    ) -> list[str]:
        return await asyncio.gather(
            *[
                self._generate(
                    min_age=min_age,
                    max_age=max_age,
                    gender=gender,
                    annual_household_income_k=annual_household_income_k,
                )
                for _ in range(count)
            ]
        )
