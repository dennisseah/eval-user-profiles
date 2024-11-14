from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from eval_user_profiles.services.user_profile_generator import UserProfileGenerator


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
                        content="```some profile```",  # noqa E501
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
async def test_user_profile_generator(mocked_azure_openai):
    generator = UserProfileGenerator(mocked_azure_openai)
    results = await generator.generate(20, 40, "female", 100, 3)
    assert 3 == len(results)


@pytest.mark.asyncio
async def test_user_profile_generator_no_result(mocked_azure_openai_no_results):
    generator = UserProfileGenerator(mocked_azure_openai_no_results)
    results = await generator.generate(20, 40, "female", 100, 3)
    assert results == ["", "", ""]
