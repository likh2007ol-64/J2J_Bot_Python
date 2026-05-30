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

START_TEXT = (
    "**Привет! Я J2J_Bot — твой помощник в изучении Java.**\n\n"
    "➤ Задавай любые вопросы по Java — я отвечу на основе всех учебников в моей библиотеке. "
    "А ещё я умею выполнять твой код и показывать правильное решение, если ошибёшься.\n\n"
    "➤ Я помогу тебе по всей программе:\n"
    "• Java Core (теория, задачи, разбор ошибок)\n"
    "• Подготовка к проекту (объясню теорию, покажу примеры кода для Git, Maven, JDBC, Hibernate, Spring)\n"
    "• Помощь на всех этапах: от первых программ до подготовки к собеседованию\n"
    "• Ответы на вопросы с указанием источников\n"
    "• Генерация правильного кода, если ошибешься\n"
    "• Разбор ошибок и помощь с отладкой\n"
    "• Вопросы и задачи для собеседований (по материалам библиотеки)\n\n"
    "➤ Всё это — в одном месте, по твоей программе.\n\n"
    "➤ Команды:\n"
    "/start — приветствие\n"
    "/help — подробная инструкция\n"
    "/about — информация о боте"
)

HELP_TEXT = (
    "📚 Как пользоваться J2J_Bot\n\n"
    "➤ ❓ Вопросы по теории\n"
    "Просто напиши свой вопрос на русском или английском. "
    "Я найду ответ в учебниках и укажу источник (автор, книга, страница).\n\n"
    "➤ 💻 Выполнение Java-кода\n"
    "Используй команду /run, а после неё напиши код в этой же строке.\n"
    "Пример: /run System.out.println(\"Hello\");\n\n"
    "⚠️ Если в коде ошибка — я сразу выдам правильное решение с объяснением.\n\n"
    "➤ 🆘 Не знаешь, как написать код? Попроси меня показать:\n"
    "/explain Напиши метод, который возвращает сумму двух чисел\n\n"
    "Если просто написать /explain — я попрошу уточнить задачу.\n\n"
    "➤ ❌ Если бот не сможет выполнить код, он объяснит причину и подскажет, что делать:\n\n"
    "• **Код слишком большой**\n"
    "  → Бот попросит разбить его на несколько маленьких частей.\n"
    "  Как сделать: запусти каждую часть отдельно командой /run.\n\n"
    "• **Программа выполняется слишком долго (возможно, бесконечный цикл)**\n"
    "  → Бот сообщит о превышении времени.\n"
    "  Что делать: проверь условия циклов, добавь счётчик или ограничение по времени внутри кода.\n\n"
    "• **Вывод программы слишком большой (более 10 000 символов)**\n"
    "  → Бот предупредит об ограничении.\n"
    "  Что сделать: сократи вывод — выводи только часть данных, используй условия или выводи результат порциями."
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

    # /start
    if lower in ("/start", "начать"):
        await bot.api.messages.send(
            peer_id=peer_id,
            message=START_TEXT,
            attachment=images.get_attachment("greeting"),
            random_id=0,
        )
        return

    # /help
    if lower == "/help":
        await bot.api.messages.send(
            peer_id=peer_id,
            message=HELP_TEXT,
            attachment=images.get_attachment("greeting"),
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

_CODE_SIZE_LIMIT = 10_000   # chars — reject before sending to JDoodle
_OUTPUT_SIZE_LIMIT = 10_000  # chars — truncate with warning
_OUTPUT_SHOW_CHARS = 500     # how many chars to show when truncating

async def _handle_run(message, bot, text: str, peer_id: int, preextracted_code: str | None = None):
    """Execute Java code via JDoodle.

    Pre-execution checks (before calling JDoodle):
      1. JDoodle not configured → friendly error
      2. Code > 10 000 chars    → ask to split

    Post-execution checks (on JDoodle result):
      3. Timeout (TLE / HTTP timeout) → advice, no DeepSeek
      4. Output > 10 000 chars        → truncate + warning
    """

    # ── 1. JDoodle configured? ────────────────────────────────────────────────
    if not JDOODLE_ENABLED:
        await _send_sad(bot, peer_id,
            "Модуль проверки кода не настроен.\n"
            "Администратор должен добавить:\n"
            "  JDODDLE_CLIENT_ID\n"
            "  JDODDLE_CLIENT_SECRET\n"
            "  JDODDLE_PLAN"
        )
        return

    # Extract code (or use pre-extracted)
    code = preextracted_code or _extract_code(text)
    if not code:
        await _send_sad(bot, peer_id,
            "Пожалуйста, отправьте код на Java после команды /run "
            "или в тройных бэктиках:\n"
            "```java\nSystem.out.println(\"Hello\");\n```"
        )
        return

    # ── 2. Code size check ────────────────────────────────────────────────────
    if len(code) > _CODE_SIZE_LIMIT:
        logger.info("Code rejected: too large (%d chars)", len(code))
        await _send_sad(bot, peer_id,
            f"Код слишком большой для выполнения в учебной среде ({len(code):,} символов).\n\n"
            "Попробуй разбить его на несколько маленьких частей "
            "и запусти каждую отдельно командой /run."
        )
        return

    await _send_thinking(bot, peer_id)

    # ── Daily limit check (free plan) ─────────────────────────────────────────
    if JDOODLE_PLAN != "pro":
        count = await database.get_jdoodle_count()
        if count >= JDOODLE_FREE_LIMIT:
            await _send_sad(bot, peer_id,
                f"Достигнут дневной лимит проверки кода ({JDOODLE_FREE_LIMIT} запросов/сутки).\n"
                "Лимит сбросится в 00:00 UTC."
            )
            return

    # ── Execute via JDoodle ───────────────────────────────────────────────────
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

    # ── 4. Timeout check ─────────────────────────────────────────────────────
    if result.get("timeout"):
        logger.info("JDoodle timeout for user code")
        await _send_sad(bot, peer_id,
            "Программа выполняется слишком долго (возможно, бесконечный цикл).\n\n"
            "Проверь условия циклов, добавь счётчик или ограничение по времени внутри кода."
        )
        return

    # Count this request (free plan)
    if JDOODLE_PLAN != "pro":
        new_count = await database.increment_jdoodle_count()
        logger.info("JDoodle daily count: %d/%d", new_count, JDOODLE_FREE_LIMIT)

    await database.log_operation("run_code", f"error={result['error']}")

    raw_output = result["output"] or ""

    # ── 5. Large output check ─────────────────────────────────────────────────
    output_too_large = len(raw_output) > _OUTPUT_SIZE_LIMIT
    if output_too_large:
        logger.info("JDoodle output truncated: %d chars", len(raw_output))
        output = raw_output[:_OUTPUT_SHOW_CHARS] + (
            f"\n\n... (вывод обрезан: показано {_OUTPUT_SHOW_CHARS} из {len(raw_output):,} символов)"
        )
        await _send_sad(bot, peer_id,
            f"Вывод программы слишком большой (более {_OUTPUT_SIZE_LIMIT:,} символов). "
            f"Показаны первые {_OUTPUT_SHOW_CHARS} символов.\n\n"
            f"{output}\n\n"
            "Попробуй сократить вывод: выводи только часть данных или используй условия."
        )
        return

    output = raw_output or "(нет вывода)"

    if not result["error"]:
        # ── Success ───────────────────────────────────────────────────────────
        await _send_happy(bot, peer_id,
            f"Код выполнен без ошибок. Молодец!\n\nВывод программы:\n{output}"
        )
        return

    # ── Error: show error + DeepSeek fix ─────────────────────────────────────
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
