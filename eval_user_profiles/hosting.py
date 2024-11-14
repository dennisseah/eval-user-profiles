"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from eval_user_profiles.protocols.i_azure_openai_service import IAzureOpenAIService
from eval_user_profiles.protocols.i_user_profile_comparator import (
    IUserProfileComparator,
)
from eval_user_profiles.protocols.i_user_profile_generator import IUserProfileGenerator

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def _() -> logging.Logger:
    return logging.getLogger("llm_operational_metrics")


@dependency_definition(container, singleton=True)
def azure_openai_service() -> IAzureOpenAIService:
    from eval_user_profiles.services.azure_openai_service import AzureOpenAIService

    return container[AzureOpenAIService]


@dependency_definition(container, singleton=True)
def user_profile_generator() -> IUserProfileGenerator:
    from eval_user_profiles.services.user_profile_generator import UserProfileGenerator

    return container[UserProfileGenerator]


@dependency_definition(container, singleton=True)
def user_profile_comparator() -> IUserProfileComparator:
    from eval_user_profiles.services.user_profile_comparator import (
        UserProfileComparator,
    )

    return container[UserProfileComparator]
