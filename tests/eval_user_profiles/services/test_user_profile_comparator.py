from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from eval_user_profiles.services.user_profile_comparator import UserProfileComparator


@pytest.fixture
def mocked_azure_openai():
    class MockedAzureOpenAI:
        async def generate(self, content, temperature):
            mocked_result = MagicMock(spec=ChatCompletion)
            mocked_result.choices = [
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='```json\n{"reasoning": "I think they are similar", "score": 0.8}\n```',  # noqa E501
                    ),
                )
            ]
            return mocked_result

        async def generate_embdding(self, text, **kwargs) -> list[float]:
            return [1.0]

    return MockedAzureOpenAI()


@pytest.fixture
def mocked_azure_openai_invalid_json():
    class MockedAzureOpenAI:
        async def generate(self, content, temperature):
            mocked_result = MagicMock(spec=ChatCompletion)
            mocked_result.choices = [
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content='```json\n{"reasoning" "I think they are similar", "score": 0.8}\n```',  # noqa E501
                    ),
                )
            ]
            return mocked_result

    return MockedAzureOpenAI()


@pytest.fixture
def mocked_azure_openai_no_results():
    class MockedAzureOpenAI:
        async def generate(self, content, temperature):
            mocked_result = MagicMock(spec=ChatCompletion)
            mocked_result.choices = []
            return mocked_result

    return MockedAzureOpenAI()


@pytest.mark.asyncio
async def test_compare(mocked_azure_openai):
    comparator = UserProfileComparator(openai_service=mocked_azure_openai)

    result = await comparator.compare("base", "profile")

    assert result is not None
    assert result.reasoning == "I think they are similar"
    assert result.score == 0.8


@pytest.mark.asyncio
async def test_compare_none(mocked_azure_openai_no_results):
    comparator = UserProfileComparator(openai_service=mocked_azure_openai_no_results)

    result = await comparator.compare("base", "profile")

    assert result is None


@pytest.mark.asyncio
async def test_compare_err(mocked_azure_openai_invalid_json):
    comparator = UserProfileComparator(openai_service=mocked_azure_openai_invalid_json)

    result = await comparator.compare("base", "profile")

    assert result is None


@pytest.mark.asyncio
async def test_compare_with_embedding(mocked_azure_openai):
    comparator = UserProfileComparator(openai_service=mocked_azure_openai)

    result = await comparator.compare_with_embedding("base", "profile")

    assert result == 1.0
