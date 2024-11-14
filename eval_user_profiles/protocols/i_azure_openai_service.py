from typing import Protocol

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)


class IAzureOpenAIService(Protocol):
    async def generate(
        self, prompts: list[ChatCompletionMessageParam], **kwargs
    ) -> ChatCompletion: ...

    async def generate_embdding(self, text: str, **kwargs) -> list[float]: ...
