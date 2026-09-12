import httpx
import pytest

from triage.ingest.github_client import fetch_issues


def test_fetch_issues_follows_pagination():
    def handler(request: httpx.Request) -> httpx.Response:
        if "page=2" in str(request.url):
            return httpx.Response(200, json=[{"number": 2}])
        return httpx.Response(
            200,
            json=[{"number": 1}],
            headers={"Link": '<https://api.github.com/x?page=2>; rel="next"'},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    issues = list(fetch_issues(client, "owner/repo"))

    assert [issue["number"] for issue in issues] == [1, 2]


def test_fetch_issues_passes_since_param():
    captured_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    list(fetch_issues(client, "owner/repo", since="2024-01-01T00:00:00Z"))

    assert "since=2024-01-01" in captured_urls[0]


def test_fetch_issues_retries_after_rate_limit(monkeypatch: pytest.MonkeyPatch):
    slept_for: list[float] = []
    monkeypatch.setattr("triage.ingest.github_client.time.sleep", slept_for.append)

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                403,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "0"},
            )
        return httpx.Response(200, json=[{"number": 1}])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    issues = list(fetch_issues(client, "owner/repo"))

    assert [issue["number"] for issue in issues] == [1]
    assert calls["count"] == 2
    assert slept_for  # backed off before retrying
