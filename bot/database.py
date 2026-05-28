import aiosqlite
import asyncio
import logging
import os
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger(__name__)


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS library_stats (
                id INTEGER PRIMARY KEY,
                book_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                last_indexed TEXT,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS operation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT,
                details TEXT,
                created_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jdoodle_limits (
                date TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
        """)
        # Insert default row if not exists
        await db.execute("""
            INSERT OR IGNORE INTO library_stats (id, book_count, chunk_count)
            VALUES (1, 0, 0)
        """)
        await db.commit()
    logger.info("Database initialized at %s", DB_PATH)


async def get_library_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT book_count, chunk_count, last_indexed FROM library_stats WHERE id=1") as cur:
            row = await cur.fetchone()
            if row:
                return {"book_count": row[0], "chunk_count": row[1], "last_indexed": row[2]}
    return {"book_count": 0, "chunk_count": 0, "last_indexed": None}


async def update_library_stats(book_count: int, chunk_count: int):
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE library_stats
            SET book_count=?, chunk_count=?, last_indexed=?, updated_at=?
            WHERE id=1
        """, (book_count, chunk_count, now, now))
        await db.commit()


async def log_operation(operation: str, details: str = ""):
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO operation_log (operation, details, created_at) VALUES (?, ?, ?)",
            (operation, details, now)
        )
        await db.commit()


def _today_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


async def get_jdoodle_count() -> int:
    """Return how many JDoodle requests have been made today (UTC)."""
    today = _today_utc()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT count FROM jdoodle_limits WHERE date=?", (today,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0


async def increment_jdoodle_count() -> int:
    """Increment today's JDoodle request counter. Returns new count."""
    today = _today_utc()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO jdoodle_limits (date, count) VALUES (?, 1) "
            "ON CONFLICT(date) DO UPDATE SET count = count + 1",
            (today,)
        )
        await db.commit()
        async with db.execute(
            "SELECT count FROM jdoodle_limits WHERE date=?", (today,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 1


async def reset_jdoodle_count() -> None:
    """Force-reset today's JDoodle counter to 0."""
    today = _today_utc()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO jdoodle_limits (date, count) VALUES (?, 0) "
            "ON CONFLICT(date) DO UPDATE SET count = 0",
            (today,)
        )
        await db.commit()
