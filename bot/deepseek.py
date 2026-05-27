import logging
import aiohttp
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL, DEEPSEEK_TEMPERATURE

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """Ты — учебный ассистент по Java для студентов JUMP2JAVA.
Отвечай ТОЛЬКО на основе предоставленных фрагментов из учебников.
Если ответ не найден во фрагментах, скажи: "Я не могу найти точный ответ в загруженных учебниках."
Отвечай на русском языке.
Обязательно укажи источник: название книги, автора, страницу (если известны).
Не используй знания, которых нет во фрагментах."""

# Sentinel error types returned instead of None for better diagnostics
ERROR_NO_BALANCE = "ERR_NO_BALANCE"
ERROR_INVALID_KEY = "ERR_INVALID_KEY"
ERROR_RATE_LIMIT = "ERR_RATE_LIMIT"
ERROR_NETWORK = "ERR_NETWORK"
ERROR_UNKNOWN = "ERR_UNKNOWN"


def _build_rag_prompt(question: str, chunks: list[dict]) -> str:
    fragments = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = f"[{i}] Книга: {meta.get('book_title', 'Неизвестно')}, Автор: {meta.get('author', 'Неизвестно')}, Стр. {meta.get('page', '?')}"
        fragments.append(f"{source}\n{chunk['text']}")

    fragments_text = "\n\n---\n\n".join(fragments)
    return f"Фрагменты из учебников:\n\n{fragments_text}\n\nВопрос студента: {question}"


async def ask_deepseek_rag(question: str, chunks: list[dict]) -> str | None:
    """Ask DeepSeek using RAG context.
    Returns answer string on success,
    or one of ERROR_* sentinel strings on specific failures,
    or None if no chunks provided.
    """
    if not chunks:
        return None

    if not DEEPSEEK_API_KEY or not DEEPSEEK_API_KEY.strip():
        logger.error("DeepSeek: DEEPSEEK_API_KEY is not set or empty")
        return ERROR_INVALID_KEY

    key_preview = DEEPSEEK_API_KEY.strip()[:8] + "..."
    prompt = _build_rag_prompt(question, chunks)

    logger.info(
        "DeepSeek request → URL: %s | model: %s | key: %s | chunks: %d | prompt_len: %d chars",
        DEEPSEEK_API_URL, DEEPSEEK_MODEL, key_preview, len(chunks), len(prompt)
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                DEEPSEEK_API_URL,
                json={
                    "model": DEEPSEEK_MODEL,
                    "temperature": DEEPSEEK_TEMPERATURE,
                    "messages": [
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2048,
                },
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY.strip()}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                status = resp.status
                body = await resp.text()

                if status == 200:
                    import json
                    data = json.loads(body)
                    answer = data["choices"][0]["message"]["content"].strip()
                    logger.info("DeepSeek response OK (%d chars)", len(answer))
                    return answer

                # Log full error response
                logger.error(
                    "DeepSeek API error | status: %d | body: %s",
                    status, body[:500]
                )

                if status == 402:
                    logger.error(
                        "DeepSeek: INSUFFICIENT BALANCE (402). "
                        "Пополните баланс на https://platform.deepseek.com/billing"
                    )
                    return ERROR_NO_BALANCE
                elif status == 401:
                    logger.error(
                        "DeepSeek: INVALID API KEY (401). "
                        "Проверьте DEEPSEEK_API_KEY в секретах Replit."
                    )
                    return ERROR_INVALID_KEY
                elif status == 429:
                    logger.error("DeepSeek: RATE LIMIT exceeded (429).")
                    return ERROR_RATE_LIMIT
                else:
                    return ERROR_UNKNOWN

    except aiohttp.ClientConnectorError as e:
        logger.error("DeepSeek: network connection error: %s", e)
        return ERROR_NETWORK
    except aiohttp.ServerTimeoutError as e:
        logger.error("DeepSeek: request timed out: %s", e)
        return ERROR_NETWORK
    except aiohttp.ClientError as e:
        logger.error("DeepSeek: client error: %s", e)
        return ERROR_NETWORK
    except Exception as e:
        logger.error("DeepSeek: unexpected error: %s", e, exc_info=True)
        return ERROR_UNKNOWN


NO_ANSWER_PHRASES = [
    "не могу найти",
    "не содержится",
    "нет в учебниках",
    "cannot find",
    "not found",
]


def is_no_answer(text: str) -> bool:
    if text in (ERROR_NO_BALANCE, ERROR_INVALID_KEY, ERROR_RATE_LIMIT, ERROR_NETWORK, ERROR_UNKNOWN):
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in NO_ANSWER_PHRASES)


def is_api_error(text: str | None) -> bool:
    return text in (ERROR_NO_BALANCE, ERROR_INVALID_KEY, ERROR_RATE_LIMIT, ERROR_NETWORK, ERROR_UNKNOWN)


def api_error_message(error_code: str) -> str:
    messages = {
        ERROR_NO_BALANCE: "⚠️ На счёте DeepSeek закончился баланс. Администратор должен пополнить аккаунт на platform.deepseek.com",
        ERROR_INVALID_KEY: "⚠️ API ключ DeepSeek недействителен. Администратор должен обновить DEEPSEEK_API_KEY.",
        ERROR_RATE_LIMIT: "⚠️ Превышен лимит запросов к DeepSeek. Попробуйте через минуту.",
        ERROR_NETWORK: "⚠️ Нет связи с сервером DeepSeek. Проверьте сеть и попробуйте позже.",
        ERROR_UNKNOWN: "⚠️ Неизвестная ошибка сервиса ответов.",
    }
    return messages.get(error_code, "⚠️ Сервис ответов временно недоступен.")
