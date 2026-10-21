"""Small bounded client for Philly's stateless MCP tools, shared by SDK and CLI.

The endpoint may be a local development server. Provider availability depends on
that server's version; this client does not implement alternate upstream paths.
"""

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

import httpx

DEFAULT_ENDPOINT = "https://philly.horv.co/mcp"
TOOLS = frozenset(
    {
        "search_datasets",
        "describe_dataset",
        "get_schema",
        "preview_dataset",
        "query_dataset",
        "aggregate_dataset",
    }
)


class RemoteToolError(RuntimeError):
    """A structured MCP failure, including a stable code when supplied."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class RemotePhilly:
    """Use typed bounded MCP operations without changing local Philly loaders.

    Calls use the server's supported stateless legacy JSON-RPC transport. No
    session, credentials, retry, redirect or automatic page traversal is used.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        url = urlsplit(endpoint)
        if (
            url.username
            or url.password
            or url.query
            or url.fragment
            or not url.hostname
            or not (
                url.scheme == "https"
                or (
                    url.scheme == "http"
                    and url.hostname in {"localhost", "127.0.0.1", "::1"}
                )
            )
        ):
            raise ValueError("Use an HTTPS MCP endpoint or loopback HTTP endpoint.")
        self.endpoint = endpoint
        self._transport = transport

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one of the six tools; return its complete structured envelope."""
        if tool not in TOOLS:
            raise ValueError("Unsupported MCP tool.")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        body = json.dumps(request, allow_nan=False).encode()
        if len(body) > 16_384:
            raise ValueError("MCP request exceeds 16 KiB.")
        async with asyncio.timeout(12):
            async with httpx.AsyncClient(
                timeout=10, follow_redirects=False, transport=self._transport
            ) as client:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RemoteToolError(
                            "HTTP_ERROR", f"MCP HTTP status {response.status_code}."
                        )
                    media_type = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if media_type not in {"application/json", "text/event-stream"}:
                        raise RemoteToolError(
                            "PROTOCOL_ERROR", "Expected MCP JSON or SSE."
                        )
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(chunks) + len(chunk) > 196_608:
                            raise RemoteToolError(
                                "LIMIT_EXCEEDED", "MCP response exceeds 192 KiB."
                            )
                        chunks.extend(chunk)
        try:
            if media_type == "text/event-stream":
                # Stateless legacy calls return one finite SSE response event.
                # Bound the complete decoded stream before parsing; never open
                # a separate GET subscription or follow event URLs.
                events = []
                for event in chunks.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
                    data = [
                        line[5:].lstrip(" ")
                        for line in event.splitlines()
                        if line.startswith("data:")
                    ]
                    if data:
                        events.append("\n".join(data))
                if len(events) != 1:
                    raise ValueError()
                envelope = json.loads(events[0])
            else:
                envelope = json.loads(chunks)
            if (
                not isinstance(envelope, dict)
                or envelope.get("jsonrpc") != "2.0"
                or envelope.get("id") != 1
            ):
                raise ValueError()
            if "error" in envelope:
                raise RemoteToolError("PROTOCOL_ERROR", "MCP request was rejected.")
            result = envelope["result"]
            value = result.get("structuredContent")
            if value is None:
                value = json.loads(result["content"][0]["text"])
            if not isinstance(value, dict):
                raise ValueError()
            if result.get("isError"):
                error = value.get("error", {})
                raise RemoteToolError(
                    str(error.get("code", "TOOL_ERROR")),
                    str(error.get("message", "MCP tool failed."))[:300],
                )
            return value
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise RemoteToolError("PROTOCOL_ERROR", "Invalid MCP response.") from exc

    async def search(self, query: str = "", **page: Any) -> dict[str, Any]:
        return await self.call("search_datasets", {"query": query, **page})

    async def describe(self, dataset_id: str, **page: Any) -> dict[str, Any]:
        return await self.call("describe_dataset", {"datasetId": dataset_id, **page})

    async def schema(
        self, dataset_id: str, resource_id: str | None = None
    ) -> dict[str, Any]:
        args = {"datasetId": dataset_id}
        if resource_id:
            args["resourceId"] = resource_id
        return await self.call("get_schema", args)

    async def query(self, dataset_id: str, **arguments: Any) -> dict[str, Any]:
        """Query with MCP names: columns, filters, limit, offset, resourceId, etc."""
        return await self.call("query_dataset", {"datasetId": dataset_id, **arguments})

    async def aggregate(self, dataset_id: str, **arguments: Any) -> dict[str, Any]:
        return await self.call(
            "aggregate_dataset", {"datasetId": dataset_id, **arguments}
        )
