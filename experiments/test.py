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
    results = await svc.generate(30, 40, "male", 100, 5)
    base = get_base_profile()
    cmp_results = await asyncio.gather(*[cmp.compare(base, user) for user in results])

    for r in filter(lambda x: x is not None, cmp_results):
        print(r.model_dump())  # type: ignore
        print()
        print()

    async def fn_embedding(user: str) -> dict[str, float | str]:
        return {
            "name": user.split("\n")[0],
            "score": await cmp.compare_with_embedding(base, user),
        }

    cmp_embedding_results = await asyncio.gather(
        *[fn_embedding(user) for user in results]
    )
    for r in cmp_embedding_results:
        print(r)
        print()
        print()


if __name__ == "__main__":
    asyncio.run(main())
