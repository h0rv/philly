import asyncio

from philly_cheesesnake import PhillyCheesesnake


async def main():
    cs = PhillyCheesesnake()

    await cs.load_all_into_tables(show_progress=True)


if __name__ == "__main__":
    asyncio.run(main())
