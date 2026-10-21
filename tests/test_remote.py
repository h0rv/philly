"""SDK/CLI transport contract, bounded I/O and failure behavior."""

import json

import httpx
import pytest

from philly import RemotePhilly, RemoteToolError
from philly.__main__ import RemoteCommands


def rpc(value, *, error=False):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "isError": error,
            "structuredContent": value,
            "content": [{"type": "text", "text": json.dumps(value)}],
        },
    }


@pytest.mark.asyncio
async def test_remote_query_preserves_typed_scope_and_sends_no_sql_or_credentials():
    result = {
        "sourceId": "noaa",
        "datasetId": "noaa:philadelphia_tides_today",
        "rows": [{"height_meters": "0.4", "datum": "MLLW"}],
        "completeness": "partial",
        "completenessReasons": ["row_limit"],
        "nextOffset": 1,
    }
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        assert "authorization" not in request.headers
        return httpx.Response(200, json=rpc(result))

    client = RemotePhilly(transport=httpx.MockTransport(handle))
    actual = await client.query(
        result["datasetId"], columns=["height_meters", "datum"], limit=1
    )
    assert actual == result
    assert seen == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_dataset",
                "arguments": {
                    "datasetId": result["datasetId"],
                    "columns": ["height_meters", "datum"],
                    "limit": 1,
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_remote_errors_keep_codes_without_silent_fallback():
    client = RemotePhilly(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json=rpc(
                    {
                        "error": {
                            "code": "UNSUPPORTED_OPERATION",
                            "message": "No filters.",
                        }
                    },
                    error=True,
                ),
            )
        )
    )
    with pytest.raises(RemoteToolError) as error:
        await client.query("loc:philadelphia_engineering", filters=[{}])
    assert error.value.code == "UNSUPPORTED_OPERATION"


@pytest.mark.asyncio
async def test_remote_rejects_redirects_malformed_oversize_and_non_json():
    for response in [
        httpx.Response(302, headers={"location": "http://localhost/secret"}),
        httpx.Response(200, json={"jsonrpc": "2.0", "id": 999, "result": {}}),
        httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {}}),
        httpx.Response(200, text="private upstream text"),
        httpx.Response(200, text="{", headers={"content-type": "application/json"}),
        httpx.Response(
            200,
            text="x" * 196_609,
            headers={"content-type": "application/json", "content-length": "1"},
        ),
    ]:
        client = RemotePhilly(transport=httpx.MockTransport(lambda _, r=response: r))
        with pytest.raises(RemoteToolError):
            await client.search()


@pytest.mark.asyncio
async def test_remote_bounds_requests_and_allows_text_only_json_fallback():
    calls = []
    value = {"datasets": []}

    def handle(request):
        calls.append(request)
        body = rpc(value)
        del body["result"]["structuredContent"]
        return httpx.Response(200, json=body)

    client = RemotePhilly(transport=httpx.MockTransport(handle))
    with pytest.raises(ValueError, match="16 KiB"):
        await client.search("x" * 17_000)
    with pytest.raises(ValueError, match="Unsupported"):
        await client.call("execute", {})
    assert calls == []
    assert await client.search("rivers") == value


def test_remote_endpoint_policy_and_cli_json_preserves_envelope(monkeypatch, capsys):
    for endpoint in [
        "http://example.com/mcp",
        "https://user:password@example.com/mcp",
        "https://example.com/mcp?token=x",
        "https://example.com/mcp#x",
        "file:///tmp/foo",
    ]:
        with pytest.raises(ValueError):
            RemotePhilly(endpoint)
    RemotePhilly("http://127.0.0.1:8787/mcp")
    seen = []

    async def call(self, tool, arguments):
        seen.append((tool, arguments))
        return {"rows": [], "completenessReasons": ["row_limit"], "nextOffset": 2}

    monkeypatch.setattr(RemotePhilly, "call", call)
    RemoteCommands().query("usgs:schuylkill_latest", columns=["value"], limit=2)
    assert seen[0][0] == "query_dataset"
    assert seen[0][1]["columns"] == ["value"]
    assert json.loads(capsys.readouterr().out)["nextOffset"] == 2


@pytest.mark.asyncio
async def test_remote_bounded_legacy_sse_and_wrappers():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content)["params"])
        data = json.dumps(rpc({"sourceId": "noaa", "rows": []}))
        return httpx.Response(
            200,
            text=f"event: message\r\ndata: {data}\r\n\r\n",
            headers={"content-type": "text/event-stream"},
        )

    client = RemotePhilly(transport=httpx.MockTransport(handle))
    for value in [
        await client.describe("noaa:philadelphia_tides_today"),
        await client.schema("noaa:philadelphia_tides_today", "0000000000000004"),
        await client.aggregate("Crime_Incidents", operation="count"),
    ]:
        assert value["sourceId"] == "noaa"
    assert calls[1]["arguments"]["resourceId"] == "0000000000000004"
    for body in ["data: {}\n\ndata: {}\n\n", "event: message\n\n"]:
        bad = RemotePhilly(
            transport=httpx.MockTransport(
                lambda _, text=body: httpx.Response(
                    200, text=text, headers={"content-type": "text/event-stream"}
                )
            )
        )
        with pytest.raises(RemoteToolError):
            await bad.search()


def test_remote_cli_retains_structured_error_code(monkeypatch, capsys):
    async def call(self, tool, arguments):
        raise RemoteToolError("BUSY", "Retry later.")

    monkeypatch.setattr(RemotePhilly, "call", call)
    with pytest.raises(SystemExit) as error:
        RemoteCommands().query("musicbrainz:philadelphia_places")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": {"code": "BUSY", "message": "Retry later."}
    }
