from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # Auth
    api_master_key: str = "change-me"

    # Data paths (all under ./data/ for clean Docker volume mounting)
    db_path: str = "./data/app.db"
    chroma_dir: str = "./data/chroma_db"
    documents_dir: str = "./data/documents"

    # Rate limiting
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # Upload
    max_file_size: int = 5 * 1024 * 1024  # 5MB

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
