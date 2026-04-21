from __future__ import annotations

from pathlib import Path

import uvicorn


def load_environment(env_file: str = ".env") -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


load_environment()


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
