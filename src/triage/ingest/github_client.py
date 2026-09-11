"""Thin client over the GitHub Issues API: pagination and rate-limit handling only.

No filtering, no reshaping — whatever the API returns (including pull requests,
which share this endpoint) is passed through as-is. Separating "issue" from
"pull request" is feature engineering and belongs in a SQL view, not here.
"""

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

API_ROOT = "https://api.github.com"
PER_PAGE = 100


def fetch_issues(
    client: httpx.Client, repo: str, since: str | None = None
) -> Iterator[dict[str, Any]]:
    """Yield every issue payload for `repo`, oldest updated first.

    `since` is an ISO-8601 timestamp: only issues updated at or after it are
    returned, which is what makes a re-pull cheap instead of a full refetch.
    """
    first_params: dict[str, Any] = {
        "state": "all",
        "per_page": PER_PAGE,
        "sort": "updated",
        "direction": "asc",
    }
    if since:
        first_params["since"] = since

    url: str | None = f"{API_ROOT}/repos/{repo}/issues"
    params: dict[str, Any] | None = first_params
    while url:
        response = _get_with_retry(client, url, params)
        yield from response.json()
        url = response.links.get("next", {}).get("url")
        params = None  # `next` URL already carries the query string


def _get_with_retry(
    client: httpx.Client, url: str, params: dict[str, Any] | None
) -> httpx.Response:
    while True:
        response = client.get(url, params=params)
        if response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0":
            _sleep_until_reset(response.headers["x-ratelimit-reset"])
            continue
        response.raise_for_status()
        return response


def _sleep_until_reset(reset_epoch: str) -> None:
    reset_at = datetime.fromtimestamp(int(reset_epoch), tz=UTC)
    wait_seconds = max((reset_at - datetime.now(UTC)).total_seconds(), 0) + 1
    time.sleep(wait_seconds)


def make_client(token: str) -> httpx.Client:
    return httpx.Client(
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
    )
