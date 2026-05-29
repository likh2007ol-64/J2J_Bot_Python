import asyncio
import json
import logging
import os
from typing import Optional

from vkbottle import Keyboard, KeyboardButtonColor, Text
from vkbottle.bot import Message

import database
import knowledge
import images
from config import (
    ADMIN_IDS, KNOWLEDGE_ROOT, CHROMA_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, STATIC_DIR,
    JDOODLE_PLAN, JDOODLE_FREE_LIMIT, JDOODLE_ENABLED,
)

logger = logging.getLogger(__name__)


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def _get_payload_cmd(message) -> str:
    """Extract 'cmd' field from button payload. Returns '' if absent or unparseable."""
    try:
        payload = getattr(message, 'payload', None)
        if not payload:
            return ''
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict):
            return str(payload.get('cmd', '')).strip().lower()
    except Exception:
        pass
    return ''


def admin_keyboard() -> Keyboard:
    kb = Keyboard(one_time=False, inline=False)
    kb.add(Text("Переиндексировать всё", payload={"cmd": "reload_knowledge"}), color=KeyboardButtonColor.PRIMARY)
    kb.add(Text("Очистить базу", payload={"cmd": "clear_knowledge"}), color=KeyboardButtonColor.NEGATIVE)
    kb.row()
    kb.add(Text("Показать файлы", payload={"cmd": "list_files"}), color=KeyboardButtonColor.SECONDARY)
    kb.add(Text("Статус библиотеки", payload={"cmd": "status"}), color=KeyboardButtonColor.SECONDARY)
    kb.row()
    kb.add(Text("JDoodle: статус", payload={"cmd": "jdoodle_status"}), color=KeyboardButtonColor.SECONDARY)
    kb.add(Text("JDoodle: сброс лимита", payload={"cmd": "reset_jdoodle_limit"}), color=KeyboardButtonColor.NEGATIVE)
    return kb


async def handle_admin_command(message: Message, bot) -> bool:
    """Handle admin commands. Returns True if handled."""
    user_id = message.from_id
    raw_text = (message.text or "").strip()
    text = raw_text.lower()
    cmd = _get_payload_cmd(message)

    logger.info(
        "handle_admin_command: user_id=%s, ADMIN_IDS=%s, text=%r, payload_cmd=%r",
        user_id, ADMIN_IDS, raw_text, cmd,
    )

    if not is_admin(user_id):
        logger.debug("User %s is not admin — skipping admin handler", user_id)
        return False

    # /admin — show keyboard panel
    if text == "/admin" or cmd == "admin":
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message="Панель администратора J2J_Bot",
            keyboard=admin_keyboard().get_json(),
            random_id=0,
        )
        return True

    # /reload_knowledge
    if text == "/reload_knowledge" or cmd == "reload_knowledge":
        await _do_reload_knowledge(message, bot)
        return True

    # /clear_knowledge
    if text == "/clear_knowledge" or cmd == "clear_knowledge":
        await _do_clear_knowledge(message, bot)
        return True

    # /list_files
    if text.startswith("/list_files") or cmd == "list_files":
        path_arg = text.replace("/list_files", "").strip()
        await _do_list_files(message, bot, path_arg)
        return True

    # /status
    if text == "/status" or cmd == "status":
        await _do_status(message, bot)
        return True

    # /jdoodle_status
    if text == "/jdoodle_status" or cmd == "jdoodle_status":
        await _do_jdoodle_status(message, bot)
        return True

    # /reset_jdoodle_limit
    if text == "/reset_jdoodle_limit" or cmd == "reset_jdoodle_limit":
        await _do_reset_jdoodle_limit(message, bot)
        return True

    # /delete_file <name>
    if text.startswith("/delete_file "):
        fname = text[len("/delete_file "):].strip()
        await _do_delete_file(message, bot, fname)
        return True

    # /delete_folder <name>
    if text.startswith("/delete_folder "):
        fname = text[len("/delete_folder "):].strip()
        await _do_delete_folder(message, bot, fname)
        return True

    return False


async def _do_reload_knowledge(message: Message, bot):
    sad = images.get_attachment("sad")
    thinking = images.get_attachment("thinking")
    happy = images.get_attachment("happy")

    await bot.api.messages.send(
        peer_id=message.peer_id,
        message="Начинаю переиндексацию книг... Это может занять несколько минут.",
        attachment=thinking,
        random_id=0,
    )

    try:
        book_count, chunk_count = await knowledge.index_knowledge_base(
            KNOWLEDGE_ROOT, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
        )
        await database.update_library_stats(book_count, chunk_count)
        await database.log_operation("reload_knowledge", f"books={book_count}, chunks={chunk_count}")

        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"Переиндексация завершена!\nКниг: {book_count}\nФрагментов: {chunk_count}",
            attachment=happy,
            random_id=0,
        )
    except Exception as e:
        logger.error("Reload knowledge error: %s", e)
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"Ошибка при индексации: {e}",
            attachment=sad,
            random_id=0,
        )


async def _do_clear_knowledge(message: Message, bot):
    sad = images.get_attachment("sad")
    try:
        await knowledge.clear_knowledge_base(CHROMA_DIR)
        await database.update_library_stats(0, 0)
        await database.log_operation("clear_knowledge")
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message="База знаний очищена.",
            random_id=0,
        )
    except Exception as e:
        logger.error("Clear knowledge error: %s", e)
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"Ошибка при очистке: {e}",
            attachment=sad,
            random_id=0,
        )


async def _do_list_files(message: Message, bot, path_arg: str):
    base = KNOWLEDGE_ROOT
    target = os.path.normpath(os.path.join(base, path_arg)) if path_arg else base

    if not target.startswith(base):
        target = base

    if not os.path.exists(target):
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"Папка не найдена: {target}",
            random_id=0,
        )
        return

    lines = [f"Папка: {target}"]
    try:
        entries = sorted(os.listdir(target))
        if not entries:
            lines.append("  (пусто)")
        for entry in entries[:50]:
            full = os.path.join(target, entry)
            if os.path.isdir(full):
                lines.append(f"  [папка] {entry}/")
            else:
                size = os.path.getsize(full)
                lines.append(f"  [файл] {entry} ({size // 1024} КБ)")
        if len(entries) > 50:
            lines.append(f"  ... и ещё {len(entries) - 50} файлов")
    except Exception as e:
        lines.append(f"Ошибка: {e}")

    await bot.api.messages.send(
        peer_id=message.peer_id,
        message="\n".join(lines),
        random_id=0,
    )


async def _do_status(message: Message, bot):
    stats = await database.get_library_stats()
    chunk_count = await knowledge.get_chunk_count(CHROMA_DIR)
    last = stats.get("last_indexed") or "никогда"
    text = (
        f"Статус библиотеки J2J_Bot\n\n"
        f"Книг проиндексировано: {stats['book_count']}\n"
        f"Фрагментов в базе: {chunk_count}\n"
        f"Последняя индексация: {last}\n"
        f"Путь к литературе: {KNOWLEDGE_ROOT}"
    )
    await bot.api.messages.send(
        peer_id=message.peer_id,
        message=text,
        random_id=0,
    )


async def _do_jdoodle_status(message: Message, bot):
    if not JDOODLE_ENABLED:
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=(
                "JDoodle: не настроен\n"
                "Добавьте в Secrets Replit:\n"
                "  JDODDLE_CLIENT_ID\n"
                "  JDODDLE_CLIENT_SECRET\n"
                "  JDODDLE_PLAN (free / pro)"
            ),
            random_id=0,
        )
        return

    count = await database.get_jdoodle_count()
    if JDOODLE_PLAN == "pro":
        limit_str = "unlimited (pro)"
        remaining = "inf"
    else:
        limit_str = str(JDOODLE_FREE_LIMIT)
        remaining = str(max(0, JDOODLE_FREE_LIMIT - count))

    text = (
        f"JDoodle статус\n\n"
        f"Тариф: {JDOODLE_PLAN}\n"
        f"Использовано сегодня: {count}\n"
        f"Лимит: {limit_str}\n"
        f"Осталось: {remaining}\n"
        f"Сброс: в 00:00 UTC"
    )
    await bot.api.messages.send(
        peer_id=message.peer_id,
        message=text,
        random_id=0,
    )


async def _do_reset_jdoodle_limit(message: Message, bot):
    await database.reset_jdoodle_count()
    await database.log_operation("reset_jdoodle_limit")
    await bot.api.messages.send(
        peer_id=message.peer_id,
        message="Счётчик запросов JDoodle сброшен до 0.",
        random_id=0,
    )


async def _do_delete_file(message: Message, bot, fname: str):
    if not fname:
        await bot.api.messages.send(peer_id=message.peer_id, message="Укажите имя файла.", random_id=0)
        return
    base = KNOWLEDGE_ROOT
    target = os.path.normpath(os.path.join(base, fname))
    if not target.startswith(base):
        await bot.api.messages.send(peer_id=message.peer_id, message="Нельзя удалить файл вне папки Литература.", random_id=0)
        return
    if not os.path.isfile(target):
        await bot.api.messages.send(peer_id=message.peer_id, message=f"Файл не найден: {fname}", random_id=0)
        return
    try:
        os.remove(target)
        await bot.api.messages.send(peer_id=message.peer_id, message=f"Файл удалён: {fname}", random_id=0)
        await database.log_operation("delete_file", fname)
    except Exception as e:
        await bot.api.messages.send(peer_id=message.peer_id, message=f"Ошибка: {e}", random_id=0)


async def _do_delete_folder(message: Message, bot, fname: str):
    if not fname:
        await bot.api.messages.send(peer_id=message.peer_id, message="Укажите имя папки.", random_id=0)
        return
    base = KNOWLEDGE_ROOT
    target = os.path.normpath(os.path.join(base, fname))
    if target == os.path.normpath(base):
        await bot.api.messages.send(peer_id=message.peer_id, message="Нельзя удалить корневую папку Литература.", random_id=0)
        return
    if not target.startswith(base):
        await bot.api.messages.send(peer_id=message.peer_id, message="Нельзя удалить папку вне Литература.", random_id=0)
        return
    if not os.path.isdir(target):
        await bot.api.messages.send(peer_id=message.peer_id, message=f"Папка не найдена: {fname}", random_id=0)
        return
    try:
        import shutil
        shutil.rmtree(target)
        await bot.api.messages.send(peer_id=message.peer_id, message=f"Папка удалена: {fname}", random_id=0)
        await database.log_operation("delete_folder", fname)
    except Exception as e:
        await bot.api.messages.send(peer_id=message.peer_id, message=f"Ошибка: {e}", random_id=0)
