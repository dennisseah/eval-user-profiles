from dataclasses import dataclass

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from lagom.environment import Env
from openai import AsyncAzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from eval_user_profiles.protocols.i_azure_openai_service import IAzureOpenAIService


class AzureOpenAIEnv(Env):
    azure_openai_endpoint: str
    azure_openai_key: str | None = None
    azure_openai_api_version: str
    azure_openai_deployed_model_name: str


@dataclass
class AzureOpenAIService(IAzureOpenAIService):
    env: AzureOpenAIEnv

    def __post_init__(self):
        if self.env.azure_openai_key is None:
            azure_credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                api_version=self.env.azure_openai_api_version,
                azure_endpoint=self.env.azure_openai_endpoint,
                azure_ad_token_provider=token_provider,
            )
        else:
            self.client = AsyncAzureOpenAI(
                api_key=self.env.azure_openai_key,
                api_version=self.env.azure_openai_api_version,
                azure_endpoint=self.env.azure_openai_endpoint,
            )

    async def generate(
        self, prompts: list[ChatCompletionMessageParam], **kwargs
    ) -> ChatCompletion:
        result: ChatCompletion = await self.client.chat.completions.create(
            model=self.env.azure_openai_deployed_model_name,
            messages=prompts,
            **kwargs,
        )

        return result
