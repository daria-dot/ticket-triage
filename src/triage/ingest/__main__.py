"""Entry point for `make ingest`: pull each configured repo, upsert into raw_issues."""

import logging

from sqlalchemy import create_engine

from triage.config import get_settings
from triage.ingest.db import last_seen_updated_at, upsert_issues
from triage.ingest.github_client import fetch_issues, make_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    client = make_client(settings.github_token)

    for repo in settings.target_repo_list:
        since = last_seen_updated_at(engine, repo)
        logger.info("ingesting %s (since=%s)", repo, since or "beginning")

        batch = []
        total = 0
        for issue in fetch_issues(client, repo, since=since):
            batch.append(issue)
            if len(batch) >= BATCH_SIZE:
                upsert_issues(engine, repo, batch)
                total += len(batch)
                batch = []
        if batch:
            upsert_issues(engine, repo, batch)
            total += len(batch)

        logger.info("ingested %s: %d issues", repo, total)


if __name__ == "__main__":
    main()
