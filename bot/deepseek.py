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


def _build_rag_prompt(question: str, chunks: list[dict]) -> str:
    fragments = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = f"[{i}] Книга: {meta.get('book_title', 'Неизвестно')}, Автор: {meta.get('author', 'Неизвестно')}, Стр. {meta.get('page', '?')}"
        fragments.append(f"{source}\n{chunk['text']}")

    fragments_text = "\n\n---\n\n".join(fragments)
    return f"Фрагменты из учебников:\n\n{fragments_text}\n\nВопрос студента: {question}"


async def ask_deepseek_rag(question: str, chunks: list[dict]) -> str:
    """Ask DeepSeek using RAG context. Returns answer string."""
    if not chunks:
        return None  # Signal: no context found

    prompt = _build_rag_prompt(question, chunks)

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
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("DeepSeek API error %d: %s", resp.status, text[:200])
                    return None
                data = await resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                return answer
    except aiohttp.ClientError as e:
        logger.error("DeepSeek connection error: %s", e)
        return None
    except Exception as e:
        logger.error("DeepSeek unexpected error: %s", e)
        return None


NO_ANSWER_PHRASES = [
    "не могу найти",
    "не содержится",
    "нет в учебниках",
    "cannot find",
    "not found",
]


def is_no_answer(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in NO_ANSWER_PHRASES)
