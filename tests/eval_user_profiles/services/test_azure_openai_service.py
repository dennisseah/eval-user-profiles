from unittest.mock import AsyncMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion
from pytest_mock import MockerFixture

from eval_user_profiles.services.azure_openai_service import (
    AzureOpenAIEnv,
    AzureOpenAIService,
)


@pytest.fixture
def mocked_azure_openai():
    mocked_azure_openai = AsyncMock()
    mocked_azure_openai.chat.completions.create.return_value = ChatCompletion(
        id="fake",
        model="fake",
        created=0,
        choices=[],
        object="chat.completion",
    )
    return mocked_azure_openai


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [None, "fake"])
async def test_generate(mocker: MockerFixture, api_key, mocked_azure_openai):
    mocker.patch(
        "eval_user_profiles.services.azure_openai_service.AsyncAzureOpenAI",
        return_value=mocked_azure_openai,
    )
    service = AzureOpenAIService(
        env=AzureOpenAIEnv(
            azure_openai_endpoint="https://api.openai.com",
            azure_openai_key=api_key,
            azure_openai_api_version="2021-09-01",
            azure_openai_deployed_model_name="fake",
        )
    )
    result = await service.generate(
        [{"role": "system", "content": "Hello there"}],
    )

    assert result is not None
