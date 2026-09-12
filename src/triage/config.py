"""Central settings object. Every value comes from the environment — see .env.example."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_token: str
    target_repos: str

    postgres_user: str = "triage"
    postgres_password: str = "changeme"
    postgres_db: str = "triage"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    mlflow_tracking_uri: str = "http://localhost:5001"
    model_version: str

    aws_region: str = "eu-west-2"

    @property
    def target_repo_list(self) -> list[str]:
        return [repo.strip() for repo in self.target_repos.split(",") if repo.strip()]

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
