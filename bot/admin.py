import asyncio
import logging
import os
from typing import Optional

from vkbottle import Keyboard, KeyboardButtonColor, Text
from vkbottle.bot import Message

import database
import knowledge
import images
from config import ADMIN_IDS, KNOWLEDGE_ROOT, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, STATIC_DIR

logger = logging.getLogger(__name__)


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def admin_keyboard() -> Keyboard:
    kb = Keyboard(one_time=False, inline=False)
    kb.add(Text("🔄 Переиндексировать всё", payload={"cmd": "reload_knowledge"}), color=KeyboardButtonColor.PRIMARY)
    kb.add(Text("🗑 Очистить базу", payload={"cmd": "clear_knowledge"}), color=KeyboardButtonColor.NEGATIVE)
    kb.row()
    kb.add(Text("📂 Показать файлы", payload={"cmd": "list_files"}), color=KeyboardButtonColor.SECONDARY)
    kb.add(Text("ℹ️ Статус библиотеки", payload={"cmd": "status"}), color=KeyboardButtonColor.SECONDARY)
    return kb


async def handle_admin_command(message: Message, bot) -> bool:
    """Handle admin commands. Returns True if handled."""
    user_id = message.from_id
    text = (message.text or "").strip().lower()

    if not is_admin(user_id):
        return False

    if text == "/admin":
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message="👨‍💼 Панель администратора J2J_Bot",
            keyboard=admin_keyboard().get_json(),
            random_id=0,
        )
        return True

    if text == "/reload_knowledge" or text == "🔄 переиндексировать всё":
        await _do_reload_knowledge(message, bot)
        return True

    if text == "/clear_knowledge" or text == "🗑 очистить базу":
        await _do_clear_knowledge(message, bot)
        return True

    if text.startswith("/list_files") or text == "📂 показать файлы":
        path_arg = text.replace("/list_files", "").strip() or ""
        await _do_list_files(message, bot, path_arg)
        return True

    if text == "/status" or text == "ℹ️ статус библиотеки":
        await _do_status(message, bot)
        return True

    if text.startswith("/delete_file "):
        fname = text[len("/delete_file "):].strip()
        await _do_delete_file(message, bot, fname)
        return True

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
        message="🔄 Начинаю переиндексацию книг... Это может занять несколько минут.",
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
            message=f"✅ Переиндексация завершена!\n📚 Книг: {book_count}\n📄 Фрагментов: {chunk_count}",
            attachment=happy,
            random_id=0,
        )
    except Exception as e:
        logger.error("Reload knowledge error: %s", e)
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"❌ Ошибка при индексации: {e}",
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
            message="🗑 База знаний очищена.",
            random_id=0,
        )
    except Exception as e:
        logger.error("Clear knowledge error: %s", e)
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"❌ Ошибка при очистке: {e}",
            attachment=sad,
            random_id=0,
        )


async def _do_list_files(message: Message, bot, path_arg: str):
    base = KNOWLEDGE_ROOT
    target = os.path.normpath(os.path.join(base, path_arg)) if path_arg else base

    # Security: don't allow going above KNOWLEDGE_ROOT
    if not target.startswith(base):
        target = base

    if not os.path.exists(target):
        await bot.api.messages.send(
            peer_id=message.peer_id,
            message=f"📂 Папка не найдена: {target}",
            random_id=0,
        )
        return

    lines = [f"📂 {target}"]
    try:
        entries = sorted(os.listdir(target))
        if not entries:
            lines.append("  (пусто)")
        for entry in entries[:50]:
            full = os.path.join(target, entry)
            if os.path.isdir(full):
                lines.append(f"  📁 {entry}/")
            else:
                size = os.path.getsize(full)
                lines.append(f"  📄 {entry} ({size // 1024} КБ)")
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
        f"ℹ️ Статус библиотеки J2J_Bot\n\n"
        f"📚 Книг проиндексировано: {stats['book_count']}\n"
        f"📄 Фрагментов в базе: {chunk_count}\n"
        f"🕐 Последняя индексация: {last}\n"
        f"📁 Путь к литературе: {KNOWLEDGE_ROOT}"
    )
    await bot.api.messages.send(
        peer_id=message.peer_id,
        message=text,
        random_id=0,
    )


async def _do_delete_file(message: Message, bot, fname: str):
    if not fname:
        await bot.api.messages.send(peer_id=message.peer_id, message="Укажите имя файла.", random_id=0)
        return
    base = KNOWLEDGE_ROOT
    target = os.path.normpath(os.path.join(base, fname))
    if not target.startswith(base):
        await bot.api.messages.send(peer_id=message.peer_id, message="❌ Нельзя удалить файл вне папки Литература.", random_id=0)
        return
    if not os.path.isfile(target):
        await bot.api.messages.send(peer_id=message.peer_id, message=f"❌ Файл не найден: {fname}", random_id=0)
        return
    try:
        os.remove(target)
        await bot.api.messages.send(peer_id=message.peer_id, message=f"✅ Файл удалён: {fname}", random_id=0)
        await database.log_operation("delete_file", fname)
    except Exception as e:
        await bot.api.messages.send(peer_id=message.peer_id, message=f"❌ Ошибка: {e}", random_id=0)


async def _do_delete_folder(message: Message, bot, fname: str):
    if not fname:
        await bot.api.messages.send(peer_id=message.peer_id, message="Укажите имя папки.", random_id=0)
        return
    base = KNOWLEDGE_ROOT
    target = os.path.normpath(os.path.join(base, fname))
    if target == os.path.normpath(base):
        await bot.api.messages.send(peer_id=message.peer_id, message="❌ Нельзя удалить корневую папку Литература.", random_id=0)
        return
    if not target.startswith(base):
        await bot.api.messages.send(peer_id=message.peer_id, message="❌ Нельзя удалить папку вне Литература.", random_id=0)
        return
    if not os.path.isdir(target):
        await bot.api.messages.send(peer_id=message.peer_id, message=f"❌ Папка не найдена: {fname}", random_id=0)
        return
    try:
        import shutil
        shutil.rmtree(target)
        await bot.api.messages.send(peer_id=message.peer_id, message=f"✅ Папка удалена: {fname}", random_id=0)
        await database.log_operation("delete_folder", fname)
    except Exception as e:
        await bot.api.messages.send(peer_id=message.peer_id, message=f"❌ Ошибка: {e}", random_id=0)
