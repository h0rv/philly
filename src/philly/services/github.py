import httpx


class GitHub:
    @staticmethod
    async def _get_file_content(url: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    @staticmethod
    def convert_app_url_to_content_url(url: str) -> str:
        return url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob", ""
        )

    @staticmethod
    async def get_all_files(
        repo: str,
        branch: str = "main",
        path: str = "",
    ) -> list[dict]:
        """
        Get all files in a GitHub repository directory.

        Args:
            repo: GitHub repository URL in format 'owner/repo'
            branch: Branch name (default: main)
            path: Directory path within repository (default: root directory)

        Returns:
            List of dictionaries containing file information
        """
        # Extract owner and repo from URL
        if repo.count("/") != 1:
            raise ValueError("Repository must be in the format 'owner/repo'")

        owner, name = repo.split("/", 1)

        # Construct GitHub API URL
        api_url = f"https://api.github.com/repos/{owner}/{name}/contents/{path}"
        if branch:
            api_url += f"?ref={branch}"

        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url)
            response.raise_for_status()

            # Return list of file content
            files = response.json()
            return files

    @staticmethod
    async def get_all_files_contents(
        repo: str,
        branch: str = "main",
        path: str = "",
    ) -> list[dict]:
        files = await GitHub.get_all_files(repo, branch, path)
        for file in files:
            content = await GitHub._get_file_content(file["download_url"])
            yield content
