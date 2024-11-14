import asyncio
from dataclasses import dataclass
from typing import Literal

from eval_user_profiles.protocols.i_azure_openai_service import IAzureOpenAIService
from eval_user_profiles.protocols.i_user_profile_generator import IUserProfileGenerator


@dataclass
class UserProfileGenerator(IUserProfileGenerator):
    openai_service: IAzureOpenAIService

    async def _generate(
        self, gender: Literal["male", "female"], annual_household_income_k: int
    ) -> str:
        result = await self.openai_service.generate(
            [
                {
                    "role": "system",
                    "content": "You are a prompt engineer. You are asked to create "
                    f"a fictional {gender} consumer of a particular brand. "
                    f"This person lives in the California State of the US, "
                    "The person is married and has an annual household income of "
                    f"{annual_household_income_k} thousand dollars. Please include "
                    "the person name, age, demographic, personality traits, and "
                    "interests.",
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
        gender: Literal["male", "female"],
        annual_household_income_k: int,
        count: int,
    ) -> list[str]:
        return await asyncio.gather(
            *[
                self._generate(
                    gender=gender, annual_household_income_k=annual_household_income_k
                )
                for _ in range(count)
            ]
        )
