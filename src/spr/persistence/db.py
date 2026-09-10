from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def make_engine(url: str = "sqlite:///spr.db"):
    if url in ("sqlite://", "sqlite:///:memory:"):
        return create_engine(url, connect_args={"check_same_thread": False},
                             poolclass=StaticPool)
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url)


# SQLAlchemy tipovi -> SQLite tipovi, za dodavanje kolona koje fale
_SQLITE_TYPES = {"VARCHAR": "TEXT", "INTEGER": "INTEGER", "FLOAT": "REAL",
                 "BOOLEAN": "INTEGER", "DATETIME": "TEXT", "JSON": "TEXT"}


def migrate_sqlite(engine) -> list[str]:
    """Dodaj kolone koje postoje u modelima, a ne i u postojećoj bazi.

    `create_all` pravi tabele koje fale, ali ne dira postojeće — pa baza sa
    ranijih pokretanja ostaje bez novih kolona i upisi pucaju. Ovo je namjerno
    minimalna migracija (samo dodavanje kolona); za više bi trebao Alembic.
    """
    if not engine.url.drivername.startswith("sqlite"):
        return []
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in have:
                    continue
                sql_type = _SQLITE_TYPES.get(
                    str(column.type).split("(")[0].upper(), "TEXT")
                conn.execute(text(
                    f'ALTER TABLE "{table.name}" '
                    f'ADD COLUMN "{column.name}" {sql_type}'))
                added.append(f"{table.name}.{column.name}")
    return added


def make_session_factory(url: str = "sqlite:///spr.db") -> sessionmaker:
    # import modela da se registruju na Base prije create_all
    from spr.persistence import models  # noqa: F401
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    migrate_sqlite(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
