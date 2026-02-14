import os
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent
for env_file in (".env", ".env.local"):
    path = _project_root / env_file
    if path.exists():
        load_dotenv(path, override=True)
_cwd = Path.cwd()
for env_file in (".env", ".env.local"):
    path = _cwd / env_file
    if path.exists() and path != _project_root / env_file:
        load_dotenv(path, override=True)


class Settings:
    _DEFAULT_URL = "postgresql://api_user:password@localhost:5432/ai_performance_coach"

    @property
    def DATABASE_URL(self) -> str:
        return os.getenv("DATABASE_URL", self._DEFAULT_URL)

    @property
    def database_url_fixed(self) -> str:
        url = self.DATABASE_URL
        if url and url.startswith("postgres://"):
            return "postgresql://" + url[len("postgres://") :]
        return url or ""


settings = Settings()
