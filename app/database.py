import ssl as _ssl
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

# Render provides DATABASE_URL as "postgresql://..." but asyncpg needs "postgresql+asyncpg://..."
database_url = settings.DATABASE_URL
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

# asyncpg doesn't support "sslmode" or "channel_binding" query params — strip them
connect_args = {}
unsupported_params = ("sslmode", "channel_binding")
if any(p in database_url for p in unsupported_params):
    parsed = urlparse(database_url)
    params = parse_qs(parsed.query)
    ssl_mode = params.pop("sslmode", [None])
    params.pop("channel_binding", None)
    if ssl_mode and ssl_mode[0] in ("require", "verify-ca", "verify-full"):
        ssl_ctx = _ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = _ssl.CERT_NONE
        connect_args["ssl"] = ssl_ctx
    new_query = urlencode({k: v[0] for k, v in params.items()})
    database_url = urlunparse(parsed._replace(query=new_query))

engine = create_async_engine(database_url, echo=False, connect_args=connect_args)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
