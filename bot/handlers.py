import asyncio
import logging
import random
import re

import images
import deepseek
import knowledge
import database
import jdoodle
import admin as adm
from config import (
    CHROMA_DIR, EMBEDDING_MODEL, RAG_TOP_K,
    THINKING_DELAY_MIN, THINKING_DELAY_MAX,
    JDOODLE_PLAN, JDOODLE_FREE_LIMIT, JDOODLE_ENABLED,
)

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Привет! Я Java-робот\n"
    "Помогаю студентам JUMP2JAVA с теорией и кодом.\n\n"
    "Команды:\n"
    "/start, /help — это меню\n"
    "/about — информация о боте\n"
    "/run <код> — выполнить Java-код\n"
    "/explain <задача> — получить эталонное решение\n\n"
    "Или просто задай вопрос по Java — отвечу по учебникам.\n\n"
    "Код можно отправить и без /run — просто вставь его в тройные бэктики:\n"
    "```java\n"
    "System.out.println(\"Hello\");\n"
    "```"
)

ABOUT_TEXT = (
    "J2J_Bot — учебный ассистент по Java\n"
    "Версия: 2.0 (теория + проверка кода)\n"
    "Сообщество: Jump2Java\n\n"
    "Отвечает на вопросы по Java на основе учебников\n"
    "Выполняет Java-код через JDoodle\n"
    "При ошибке — показывает эталон через DeepSeek AI\n"
    "Технологии: RAG + DeepSeek AI + ChromaDB + JDoodle"
)

# Regex to extract ```java ... ``` or ``` ... ``` code blocks
# Handles both Unix (\n) and Windows (\r\n) line endings
_CODE_BLOCK_RE = re.compile(
    r"```(?:java)?\s*[\r\n]+(.*?)[\r\n]*```",
    re.DOTALL | re.IGNORECASE,
)

# Fallback: single-line code block  ```java code ```
_CODE_BLOCK_INLINE_RE = re.compile(
    r"```(?:java)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    """Normalize VK message text: replace <br> variants with newlines, strip excess whitespace."""
    # VK sometimes replaces newlines with <br> tags
    t = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # Collapse multiple blank lines to one
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


# Keywords that indicate Java code even without backtick fences
_JAVA_CODE_HINTS = re.compile(
    r'\b(public\s+class|public\s+static\s+void\s+main|System\.out\.print|'
    r'import\s+java\.|class\s+\w+\s*\{|new\s+\w+\s*\()',
    re.IGNORECASE,
)


def _looks_like_java_code(text: str) -> bool:
    """Heuristic: does this text look like Java source code?"""
    # Must have both braces and at least one Java keyword
    has_braces = '{' in text and '}' in text
    has_keyword = bool(_JAVA_CODE_HINTS.search(text))
    return has_braces and has_keyword


def _extract_code(text: str) -> str | None:
    """Extract Java code from message text.

    Priority:
    1. /run <code> — everything after the command
    2. ```java\\n...\\n``` block (multi-line)
    3. ```...``` block (inline / no language tag)
    Returns None if no code found.
    """
    stripped = text.strip()

    # /run command
    if stripped.lower().startswith("/run"):
        code = stripped[4:].strip()
        if code:
            return code

    # Multi-line triple-backtick block (primary)
    match = _CODE_BLOCK_RE.search(stripped)
    if match:
        code = match.group(1).strip()
        if code:
            return code

    # Inline / compact block (fallback)
    match = _CODE_BLOCK_INLINE_RE.search(stripped)
    if match:
        code = match.group(1).strip()
        if code:
            return code

    return None


def _has_code_block(text: str) -> bool:
    """Return True if the message contains a triple-backtick code block."""
    return "```" in text


async def _send_thinking(bot, peer_id: int):
    thinking_attach = images.get_attachment("thinking")
    await bot.api.messages.send(
        peer_id=peer_id,
        message="Секунду, думаю...",
        attachment=thinking_attach,
        random_id=0,
    )
    delay = random.uniform(THINKING_DELAY_MIN, THINKING_DELAY_MAX)
    await asyncio.sleep(delay)


async def _send_sad(bot, peer_id: int, text: str):
    await bot.api.messages.send(
        peer_id=peer_id,
        message=text,
        attachment=images.get_attachment("sad"),
        random_id=0,
    )


async def _send_happy(bot, peer_id: int, text: str):
    await bot.api.messages.send(
        peer_id=peer_id,
        message=text,
        attachment=images.get_attachment("happy"),
        random_id=0,
    )


async def handle_message(message, bot):
    # Normalize first: replace VK's <br> line-break substitutions with real newlines
    raw_text = (message.text or "").strip()
    text = _normalize_text(raw_text)
    peer_id = message.peer_id
    user_id = message.from_id
    lower = text.lower()

    # Log every incoming message to help debug routing
    logger.info(
        "MSG from_id=%s peer_id=%s is_admin=%s text=%r",
        user_id, peer_id, adm.is_admin(user_id), text[:150],
    )
    if raw_text != text:
        logger.info("Text normalized (had <br> tags): %r -> %r", raw_text[:80], text[:80])

    # Admin commands first (checked by text AND payload)
    if adm.is_admin(user_id):
        handled = await adm.handle_admin_command(message, bot)
        if handled:
            return

    # /start or /help
    if lower in ("/start", "/help", "начать"):
        greeting_attach = images.get_attachment("greeting")
        await bot.api.messages.send(
            peer_id=peer_id,
            message=HELP_TEXT,
            attachment=greeting_attach,
            random_id=0,
        )
        return

    # /about
    if lower == "/about":
        await bot.api.messages.send(
            peer_id=peer_id,
            message=ABOUT_TEXT,
            random_id=0,
        )
        return

    # /run — execute Java code
    if lower.startswith("/run"):
        await _handle_run(message, bot, text, peer_id)
        return

    # /explain — reference solution by task
    if lower.startswith("/explain"):
        await _handle_explain(message, bot, text, peer_id)
        return

    # Code block in message (without /run command): ```java ... ``` or ``` ... ```
    if _has_code_block(text) and not lower.startswith("/"):
        code = _extract_code(text)
        if code:
            logger.info("Auto-detected backtick code block from user %s", user_id)
            await _handle_run(message, bot, text, peer_id)
            return

    # Heuristic: message starts with "java" prefix (e.g. "java<br>public class...")
    # or contains recognizable Java constructs without backticks
    if not lower.startswith("/"):
        # Strip a leading "java" word that VK prepends when the user types ```java
        # and VK collapses the fence into plain text
        java_stripped = re.sub(r'^java\s*', '', text, flags=re.IGNORECASE).strip()
        if _looks_like_java_code(java_stripped):
            logger.info("Auto-detected bare Java code (java-prefix heuristic) from user %s", user_id)
            await _handle_run(message, bot, text, peer_id, preextracted_code=java_stripped)
            return
        if _looks_like_java_code(text):
            logger.info("Auto-detected bare Java code (no-prefix heuristic) from user %s", user_id)
            await _handle_run(message, bot, text, peer_id, preextracted_code=text)
            return

    # Unknown slash command
    if text.startswith("/"):
        await bot.api.messages.send(
            peer_id=peer_id,
            message="Неизвестная команда. Напишите /help для справки.",
            random_id=0,
        )
        return

    # Empty message
    if not text:
        return

    # Theory question — RAG pipeline
    await _handle_question(message, bot, text, peer_id)


# ─── /run handler ────────────────────────────────────────────────────────────

async def _handle_run(message, bot, text: str, peer_id: int, preextracted_code: str | None = None):
    """Execute Java code via JDoodle.

    If preextracted_code is given it is used directly; otherwise code is
    extracted from text via _extract_code().
    """

    # Check if JDoodle is configured
    if not JDOODLE_ENABLED:
        await _send_sad(bot, peer_id,
            "Модуль проверки кода не настроен.\n"
            "Администратор должен добавить:\n"
            "  JDODDLE_CLIENT_ID\n"
            "  JDODDLE_CLIENT_SECRET\n"
            "  JDODDLE_PLAN"
        )
        return

    # Use pre-extracted code if supplied, otherwise extract from text
    code = preextracted_code or _extract_code(text)
    if not code:
        await _send_sad(bot, peer_id,
            "Пожалуйста, отправьте код на Java после команды /run "
            "или в тройных бэктиках:\n"
            "```java\nSystem.out.println(\"Hello\");\n```"
        )
        return

    await _send_thinking(bot, peer_id)

    # Check daily limit (free plan only)
    if JDOODLE_PLAN != "pro":
        count = await database.get_jdoodle_count()
        if count >= JDOODLE_FREE_LIMIT:
            await _send_sad(bot, peer_id,
                f"Достигнут дневной лимит проверки кода ({JDOODLE_FREE_LIMIT} запросов/сутки).\n"
                "Лимит сбросится в 00:00 UTC."
            )
            return

    # Execute code
    result = await jdoodle.execute_java_code(code)

    if result["service_error"]:
        await _send_sad(bot, peer_id,
            "Сервис выполнения кода временно недоступен. Попробуйте позже."
        )
        return

    if result["limit_exceeded"]:
        await _send_sad(bot, peer_id,
            f"Достигнут дневной лимит проверки кода ({JDOODLE_FREE_LIMIT} запросов/сутки).\n"
            "Лимит сбросится в 00:00 UTC."
        )
        return

    # Count this request (free plan)
    if JDOODLE_PLAN != "pro":
        new_count = await database.increment_jdoodle_count()
        logger.info("JDoodle daily count: %d/%d", new_count, JDOODLE_FREE_LIMIT)

    await database.log_operation("run_code", f"error={result['error']}")

    output = result["output"] or "(нет вывода)"
    if len(output) > 1500:
        output = output[:1500] + "\n...(вывод обрезан)"

    if not result["error"]:
        # Success
        await _send_happy(bot, peer_id,
            f"Код выполнен без ошибок. Молодец!\n\nВывод программы:\n{output}"
        )
        return

    # Error — send error message, then generate fix via DeepSeek
    await bot.api.messages.send(
        peer_id=peer_id,
        message=(
            f"Ошибка:\n{output}\n\n"
            "Ниже — правильное решение с объяснением, чтобы ты мог разобраться."
        ),
        attachment=images.get_attachment("sad"),
        random_id=0,
    )

    fix = await deepseek.ask_deepseek_fix(code, output)

    if deepseek.is_api_error(fix):
        logger.error("DeepSeek fix failed: %s", fix)
        await _send_sad(bot, peer_id,
            "Не удалось сгенерировать эталонное решение. "
            "Источник временно не доступен."
        )
        return

    if fix:
        if len(fix) > 4000:
            fix = fix[:3990] + "\n...(обрезано)"
        await bot.api.messages.send(
            peer_id=peer_id,
            message=fix,
            attachment=images.get_attachment("happy"),
            random_id=0,
        )


# ─── /explain handler ────────────────────────────────────────────────────────

async def _handle_explain(message, bot, text: str, peer_id: int):
    """Generate reference Java solution for a given task."""
    task = text.strip()
    # Strip /explain prefix
    if task.lower().startswith("/explain"):
        task = task[8:].strip()

    if not task:
        await bot.api.messages.send(
            peer_id=peer_id,
            message=(
                "Для какой задачи нужно объяснение?\n"
                "Напишите условие после команды, например:\n"
                "/explain Написать метод, который переворачивает строку"
            ),
            random_id=0,
        )
        return

    await _send_thinking(bot, peer_id)

    answer = await deepseek.ask_deepseek_explain(task)

    if deepseek.is_api_error(answer):
        logger.error("DeepSeek explain failed: %s", answer)
        await _send_sad(bot, peer_id, "Источник временно не доступен.")
        return

    if not answer:
        await _send_sad(bot, peer_id, "Не удалось сформировать решение. Попробуйте позже.")
        return

    if len(answer) > 4000:
        answer = answer[:3990] + "\n...(обрезано)"

    await bot.api.messages.send(
        peer_id=peer_id,
        message=answer,
        attachment=images.get_attachment("happy"),
        random_id=0,
    )


# ─── Theory Q&A (RAG) ────────────────────────────────────────────────────────

async def _handle_question(message, bot, question: str, peer_id: int):
    """Handle a theory question using RAG."""
    await _send_thinking(bot, peer_id)

    try:
        chunk_count = await knowledge.get_chunk_count(CHROMA_DIR)
        if chunk_count == 0:
            await _send_sad(bot, peer_id,
                "База знаний ещё не загружена. Администратор должен выполнить /reload_knowledge."
            )
            return

        chunks = await knowledge.search_knowledge(question, CHROMA_DIR, EMBEDDING_MODEL, RAG_TOP_K)

        if not chunks:
            await _send_sad(bot, peer_id,
                "Я не могу найти точный ответ в загруженных учебниках. "
                "Попробуйте переформулировать вопрос."
            )
            return

        answer = await deepseek.ask_deepseek_rag(question, chunks)

        if answer is None:
            await _send_sad(bot, peer_id, "Источник временно не доступен.")
            return

        if deepseek.is_api_error(answer):
            logger.error("DeepSeek RAG error: %s", answer)
            await _send_sad(bot, peer_id, "Источник временно не доступен.")
            return

        if deepseek.is_no_answer(answer):
            await _send_sad(bot, peer_id,
                "Я не могу найти точный ответ в загруженных учебниках. "
                "Попробуйте переформулировать вопрос."
            )
            return

        if len(answer) > 4000:
            answer = answer[:3990] + "\n...(ответ обрезан)"

        await _send_happy(bot, peer_id, answer)

    except Exception as e:
        logger.error("Error handling question: %s", e, exc_info=True)
        await _send_sad(bot, peer_id, "Произошла внутренняя ошибка. Попробуйте позже.")
