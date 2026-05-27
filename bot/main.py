import asyncio
import logging
import os
import sys

# Add bot directory to path
sys.path.insert(0, os.path.dirname(__file__))

from vkbottle.bot import Bot, Message

import database
import images
import handlers
from config import VK_TOKEN, STATIC_DIR, KNOWLEDGE_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def main():
    if not VK_TOKEN:
        logger.error("VK_TOKEN is not set!")
        sys.exit(1)

    bot = Bot(token=VK_TOKEN)

    @bot.on.message()
    async def message_handler(message: Message):
        try:
            await handlers.handle_message(message, bot)
        except Exception as e:
            logger.error("Unhandled error in message_handler: %s", e, exc_info=True)

    # Init DB and images before polling starts
    logger.info("=== J2J_Bot starting (Phase 1 — Theory) ===")
    await database.init_db()
    images.ensure_static_images(STATIC_DIR)

    # Try to upload images; skip gracefully on error
    try:
        logger.info("Uploading robot images to VK...")
        await images.upload_images_to_vk(bot, STATIC_DIR)
    except Exception as e:
        logger.warning("Could not upload images to VK (will send text-only): %s", e)

    if os.path.exists(KNOWLEDGE_ROOT):
        logger.info("Knowledge root found: %s", KNOWLEDGE_ROOT)
    else:
        logger.warning(
            "Knowledge root not found: %s — create this folder and add PDFs, then run /reload_knowledge",
            KNOWLEDGE_ROOT,
        )

    print(
        "\n✅ Фаза 1 бота J2J_Bot (Python) запущена.\n"
        "   - Реализованы: RAG (ChromaDB + эмбеддинги + DeepSeek), управление библиотекой, визуальный персонаж.\n"
        "   - Команды /run и /explain отключены (заглушка).\n"
        "   - Для развёртывания на BotHost:\n"
        "       1. Установите переменные окружения (VK_TOKEN, DEEPSEEK_API_KEY, ADMIN_IDS).\n"
        "       2. Включите общее хранилище, загрузите папку Литература в /app/shared/.\n"
        "       3. Выполните /reload_knowledge после запуска.\n"
        "   - Замените файлы в static/ на реальные PNG-изображения робота.\n"
    )

    logger.info("Starting Long Poll listener...")

    # Use the loop_wrapper approach that vkbottle recommends
    # run_polling() is designed for use inside an already-running loop
    await bot.run_polling()


if __name__ == "__main__":
    # nest_asyncio is needed if running inside an already-running loop (e.g. Jupyter/Replit)
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    asyncio.run(main())
