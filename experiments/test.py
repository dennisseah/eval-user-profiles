import asyncio
import os

from eval_user_profiles.hosting import container
from eval_user_profiles.protocols.i_user_profile_comparator import (
    IUserProfileComparator,
)
from eval_user_profiles.protocols.i_user_profile_generator import IUserProfileGenerator


def get_base_profile() -> str:
    with open(os.path.join("data", "base_profile.txt"), "r") as f:
        return f.read()


async def main():
    svc = container[IUserProfileGenerator]
    cmp = container[IUserProfileComparator]

    # create 5 users with annual household income of 100k
    results = await svc.generate("male", 100, 5)
    base = get_base_profile()
    cmp_results = await asyncio.gather(*[cmp.compare(base, user) for user in results])

    for r in filter(lambda x: x is not None, cmp_results):
        print(r.model_dump())  # type: ignore
        print()
        print()


if __name__ == "__main__":
    asyncio.run(main())
