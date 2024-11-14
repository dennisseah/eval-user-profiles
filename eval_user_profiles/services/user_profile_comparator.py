import json
from dataclasses import dataclass

from eval_user_profiles.models.comparison_result import ComparisonResult
from eval_user_profiles.protocols.i_azure_openai_service import IAzureOpenAIService
from eval_user_profiles.protocols.i_user_profile_comparator import (
    IUserProfileComparator,
)


@dataclass
class UserProfileComparator(IUserProfileComparator):
    openai_service: IAzureOpenAIService

    async def compare(self, base: str, profile: str) -> ComparisonResult | None:
        result = await self.openai_service.generate(
            [
                {
                    "role": "system",
                    "content": "You are a product consumer expert. You are asked to "
                    f"compare two consumer profiles. The first profile is: \n{base}\n\n"
                    f"The second profile is: \n{profile}\n\n"
                    "Score the second profile based on how similar it is to the first "
                    "profile. The score ranges from 0 to 1, where 0 means the two "
                    "profiles are completely different and 1 means the two profiles "
                    "are identical. When comparing them, you shall put more weights on "
                    "where they live, their annual household income, and martial "
                    "status. You can also consider their age."
                    "\nProvide your score in this a JSON format as "
                    "follows: "
                    + """
                    {
                        "reasoning": "<your reasoning>",
                        "score": <your score>
                    }
                    """,
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
