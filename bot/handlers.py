import asyncio
import logging
import random

import images
import deepseek
import knowledge
import admin as adm
from config import CHROMA_DIR, EMBEDDING_MODEL, RAG_TOP_K, THINKING_DELAY_MIN, THINKING_DELAY_MAX

logger = logging.getLogger(__name__)

STUB_COMMANDS = {"/run", "/explain"}

HELP_TEXT = (
    "Привет! Я Java-робот 🤖\n"
    "Помогаю студентам JUMP2JAVA с теорией.\n"
    "Спроси меня о Java — я отвечу на основе учебников.\n\n"
    "Команды:\n"
    "/start, /help — это меню\n"
    "/about — информация о боте\n\n"
    "(Проверка кода появится в следующей версии)"
)

ABOUT_TEXT = (
    "🤖 J2J_Bot — учебный ассистент по Java\n"
    "Версия: 1.0 (теория)\n"
    "Сообщество: Jump2Java\n\n"
    "Отвечает на вопросы по Java на основе загруженных учебников.\n"
    "Технологии: RAG + DeepSeek AI + ChromaDB"
)


def _build_fallback_answer(question: str, chunks: list[dict], error_msg: str) -> str:
    """Build a fallback answer from RAG chunks when DeepSeek is unavailable."""
    lines = [
        error_msg,
        "",
        f"Тем не менее, вот что я нашёл в учебниках по запросу «{question}»:",
        "",
    ]
    for i, chunk in enumerate(chunks[:3], 1):
        meta = chunk.get("metadata", {})
        title = meta.get("book_title", "Неизвестно")
        page = meta.get("page", "?")
        text = chunk["text"]
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"📖 [{i}] {title}, стр. {page}:")
        lines.append(text)
        lines.append("")
    result = "\n".join(lines).strip()
    if len(result) > 4000:
        result = result[:3990] + "\n…(обрезано)"
    return result


async def _send_thinking(bot, peer_id: int):
    thinking_attach = images.get_attachment("thinking")
    await bot.api.messages.send(
        peer_id=peer_id,
        message="Секунду, думаю… 🤔",
        attachment=thinking_attach,
        random_id=0,
    )
    delay = random.uniform(THINKING_DELAY_MIN, THINKING_DELAY_MAX)
    await asyncio.sleep(delay)


async def handle_message(message, bot):
    text = (message.text or "").strip()
    peer_id = message.peer_id
    user_id = message.from_id
    lower = text.lower()

    # Check admin commands first
    if adm.is_admin(user_id):
        handled = await adm.handle_admin_command(message, bot)
        if handled:
            return

    # Stub for /run and /explain
    if lower in STUB_COMMANDS or lower.startswith("/run "):
        await bot.api.messages.send(
            peer_id=peer_id,
            message="Функция проверки кода будет добавлена в следующей версии. Пока я могу отвечать только на теоретические вопросы по Java.",
            random_id=0,
        )
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

    # Skip empty messages and commands we don't recognize starting with /
    if not text or (text.startswith("/") and lower not in ("/start", "/help", "/about")):
        if text.startswith("/"):
            await bot.api.messages.send(
                peer_id=peer_id,
                message="Неизвестная команда. Напишите /help для справки.",
                random_id=0,
            )
        return

    # Treat as a question — RAG pipeline
    await _handle_question(message, bot, text, peer_id)


async def _handle_question(message, bot, question: str, peer_id: int):
    """Handle a theory question using RAG."""
    # Send thinking indicator
    await _send_thinking(bot, peer_id)

    sad_attach = images.get_attachment("sad")
    happy_attach = images.get_attachment("happy")

    try:
        # Check if knowledge base is empty
        chunk_count = await knowledge.get_chunk_count(CHROMA_DIR)
        if chunk_count == 0:
            await bot.api.messages.send(
                peer_id=peer_id,
                message="База знаний ещё не загружена. Администратор должен выполнить /reload_knowledge.",
                attachment=sad_attach,
                random_id=0,
            )
            return

        # Search for relevant chunks
        chunks = await knowledge.search_knowledge(question, CHROMA_DIR, EMBEDDING_MODEL, RAG_TOP_K)

        if not chunks:
            await bot.api.messages.send(
                peer_id=peer_id,
                message="Я не могу найти точный ответ в загруженных учебниках. Попробуйте переформулировать вопрос.",
                attachment=sad_attach,
                random_id=0,
            )
            return

        # Ask DeepSeek
        answer = await deepseek.ask_deepseek_rag(question, chunks)

        if answer is None:
            # Should not happen (chunks were non-empty), but guard anyway
            await bot.api.messages.send(
                peer_id=peer_id,
                message="Сервис ответов временно недоступен, попробуйте позже.",
                attachment=sad_attach,
                random_id=0,
            )
            return

        if deepseek.is_api_error(answer):
            # DeepSeek unavailable — show RAG chunks as fallback
            error_msg = deepseek.api_error_message(answer)
            fallback = _build_fallback_answer(question, chunks, error_msg)
            await bot.api.messages.send(
                peer_id=peer_id,
                message=fallback,
                attachment=sad_attach,
                random_id=0,
            )
            return

        if deepseek.is_no_answer(answer):
            await bot.api.messages.send(
                peer_id=peer_id,
                message=f"Я не могу найти точный ответ в загруженных учебниках. Попробуйте переформулировать вопрос.\n\n{answer}",
                attachment=sad_attach,
                random_id=0,
            )
            return

        # Successful answer — trim if too long for VK (max 4096 chars)
        if len(answer) > 4000:
            answer = answer[:3990] + "\n...(ответ обрезан)"

        await bot.api.messages.send(
            peer_id=peer_id,
            message=answer,
            attachment=happy_attach,
            random_id=0,
        )

    except Exception as e:
        logger.error("Error handling question: %s", e)
        await bot.api.messages.send(
            peer_id=peer_id,
            message="Произошла внутренняя ошибка. Попробуйте позже.",
            attachment=sad_attach,
            random_id=0,
        )
