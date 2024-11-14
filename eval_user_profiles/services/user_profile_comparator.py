import json
from dataclasses import dataclass

import numpy as np

from eval_user_profiles.models.comparison_result import ComparisonResult
from eval_user_profiles.protocols.i_azure_openai_service import IAzureOpenAIService
from eval_user_profiles.protocols.i_user_profile_comparator import (
    IUserProfileComparator,
)

PROMPT = """You are a product consumer expert. You are asked to compare a profile against a desired profile.

The desired user profile is: 
{base}

The provided profile is:
{profile}

Score the provided profile based on how similar it is to the desired profile.
The score ranges from 0 to 1, where 0 means that you will never consider the provided profile for product testing and 1 means that you will definitely consider it.

When comparing them, you shall consider the following aspects:

1. Gender of the person. Having the right gender is important for product testing.
2. State where the person live. The product is only available in certain states hence the person must live in the right state.
3. Marital status
4. Annual household income
5. Personality Traits


Provide your score in this a JSON format as follows:

```json
{
    "reasoning": "<your reasoning>",
    "score": <your score (float type}>
}```
"""  # noqa E501


@dataclass
class UserProfileComparator(IUserProfileComparator):
    openai_service: IAzureOpenAIService

    async def compare(self, base: str, profile: str) -> ComparisonResult | None:
        result = await self.openai_service.generate(
            [
                {
                    "role": "system",
                    "content": PROMPT.replace("{base}", base).replace(
                        "{profile}", profile
                    ),
                }
            ],
            temperature=0.5,
        )

        content = result.choices[0].message.content if result.choices else None
        if not content:
            return None

        if content.startswith("```json"):
            content = content[7:-3]

        try:
            return ComparisonResult(**json.loads(content))
        except json.JSONDecodeError:
            print(result.choices[0].message.content)
            return None

    async def compare_with_embedding(self, base: str, profile: str) -> float:
        embedding_base = await self.openai_service.generate_embdding(base)
        embedding_profile = await self.openai_service.generate_embdding(profile)
        dot_product = np.dot(embedding_base, embedding_profile)
        magnitude_1 = np.linalg.norm(embedding_base)
        magnitude_2 = np.linalg.norm(embedding_profile)
        return float(dot_product / (magnitude_1 * magnitude_2))
