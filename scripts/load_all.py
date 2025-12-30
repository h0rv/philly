import asyncio

from philly import Philly


async def main():
    cs = Philly()

    await cs.load_all(show_progress=True)


if __name__ == "__main__":
    asyncio.run(main())
