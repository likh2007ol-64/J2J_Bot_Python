import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vkbottle.bot import Bot, Message
from vkbottle.polling import BotPolling

import database
import images
import handlers
from config import VK_TOKEN, VK_GROUP_ID, STATIC_DIR, KNOWLEDGE_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def check_token(api) -> bool:
    """Verify token has required permissions."""
    try:
        resp = await api.groups.get_long_poll_settings(group_id=VK_GROUP_ID)
        logger.info("Long Poll settings: enabled=%s", resp.is_enabled)
        return True
    except Exception as e:
        code = getattr(e, 'code', None)
        logger.error("Token check failed (VK error %s): %s", code, e)
        if code == 38:
            logger.error(
                "\n"
                "══════════════════════════════════════════════════════\n"
                "  ОШИБКА: Токен VK не имеет нужных прав (ошибка 38)\n"
                "══════════════════════════════════════════════════════\n"
                "  Пожалуйста, проверьте в настройках сообщества ВК:\n"
                "  1. Управление → Работа с API → Создать/обновить ключ\n"
                "  2. Убедитесь что включены права:\n"
                "     ✓ Управление сообществом\n"
                "     ✓ Управление сообщениями\n"
                "  3. Работа с API → Long Poll API → Включить\n"
                "     Версия API: 5.131 или выше\n"
                "  4. Скопируйте новый токен и обновите VK_TOKEN в секретах.\n"
                "══════════════════════════════════════════════════════\n"
            )
        return False


def main():
    if not VK_TOKEN:
        logger.error("VK_TOKEN is not set!")
        sys.exit(1)
    if not VK_GROUP_ID:
        logger.error("VK_GROUP_ID is not set!")
        sys.exit(1)

    polling = BotPolling(group_id=VK_GROUP_ID)
    bot = Bot(token=VK_TOKEN, polling=polling)

    @bot.on.message()
    async def message_handler(message: Message):
        try:
            await handlers.handle_message(message, bot)
        except Exception as e:
            logger.error("Unhandled error in message_handler: %s", e, exc_info=True)

    async def startup():
        logger.info("=== J2J_Bot starting (Phase 1 — Theory) ===")
        logger.info("Group ID: %s", VK_GROUP_ID)

        # Check token permissions
        ok = await check_token(bot.api)
        if not ok:
            logger.warning("Token check failed — bot may not receive messages. See instructions above.")

        await database.init_db()
        images.ensure_static_images(STATIC_DIR)

        logger.info("Uploading robot images to VK...")
        await images.upload_images_to_vk(bot, STATIC_DIR)

        if os.path.exists(KNOWLEDGE_ROOT):
            logger.info("Knowledge root: %s", KNOWLEDGE_ROOT)
        else:
            logger.warning("Knowledge root not found: %s — add PDFs and run /reload_knowledge", KNOWLEDGE_ROOT)

        logger.info("Bot ready. Listening for messages...")
        print(
            "\n✅ Фаза 1 бота J2J_Bot (Python) запущена.\n"
            "   - RAG (ChromaDB + эмбеддинги + DeepSeek), управление библиотекой, визуальный персонаж.\n"
            "   - /run и /explain — заглушка.\n"
            "   - Для BotHost: задайте VK_TOKEN, DEEPSEEK_API_KEY, ADMIN_IDS, VK_GROUP_ID;\n"
            "     загрузите папку Литература в /app/shared/; выполните /reload_knowledge.\n"
            "   - Замените static/*.png на реальные PNG-изображения робота.\n"
        )

    bot.loop_wrapper.add_task(startup())
    bot.run_forever()


if __name__ == "__main__":
    main()
