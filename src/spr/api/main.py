from spr.api.app import create_app
from spr.engine.factory import build_default_engine
from spr.persistence.db import make_session_factory

# Trenira model na startu; pokretati sa: uvicorn spr.api.main:app
app = create_app(
    build_default_engine(),
    session_factory=make_session_factory("sqlite:///spr.db"),
    artifacts_dir="artifacts",
)
