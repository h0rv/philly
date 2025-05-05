import asyncio

from cheesesnake import Cheesesnake


async def main():
    cs = Cheesesnake()

    await cs.load_all_into_tables(show_progress=True)


if __name__ == "__main__":
    asyncio.run(main())
