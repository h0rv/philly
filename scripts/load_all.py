import asyncio

from philly import Philly


async def main():
    cs = Philly()

    await cs.load_all_into_tables(show_progress=True)


if __name__ == "__main__":
    asyncio.run(main())
