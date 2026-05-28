import asyncio
import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_attachments: dict[str, str] = {}

ROBOT_IMAGES = {
    "greeting": "robot_greeting.png",
    "thinking": "robot_thinking.png",
    "happy":    "robot_happy.png",
    "sad":      "robot_sad.png",
}


def ensure_static_images(static_dir: str):
    """Create proper placeholder images via Pillow if they don't exist."""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(static_dir, exist_ok=True)

    placeholders = {
        "robot_greeting.png": ((70,  130, 180), "👋"),
        "robot_thinking.png": ((255, 165,   0), "🤔"),
        "robot_happy.png":    ((50,  205,  50), "✅"),
        "robot_sad.png":      ((220,  20,  60), "😕"),
    }

    for fname, (color, emoji) in placeholders.items():
        fpath = os.path.join(static_dir, fname)
        if not os.path.exists(fpath):
            img = Image.new("RGB", (256, 256), color=color)
            draw = ImageDraw.Draw(img)
            # Draw a simple label in the center
            text = f"J2J\n{emoji}"
            draw.text((128, 128), text, fill=(255, 255, 255), anchor="mm")
            img.save(fpath, format="PNG")
            logger.info("Created Pillow placeholder: %s (%dx%d)", fpath, *img.size)


def _prepare_image_buffer(image_path: str, max_size: int = 256) -> io.BytesIO:
    """
    Open image, resize to max_size keeping aspect ratio,
    save as JPEG (quality=90) into a BytesIO buffer.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    new_w, new_h = img.size

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    logger.info(
        "Prepared %s: orig=%dx%d → %dx%d, JPEG size=%d bytes",
        os.path.basename(image_path), orig_w, orig_h, new_w, new_h, buf.getbuffer().nbytes,
    )
    return buf


async def _upload_single_image(bot, image_path: str) -> Optional[str]:
    """Upload a single image to VK messages. Returns 'photo{owner_id}_{id}' or None."""
    fname = os.path.basename(image_path)
    try:
        import aiohttp
        api = bot.api

        # Step 1 — get upload server
        upload_server_resp = await api.photos.get_messages_upload_server(peer_id=0)
        upload_url = upload_server_resp.upload_url
        logger.info("[%s] Upload server URL: %s", fname, upload_url)

        # Step 2 — prepare resized JPEG buffer (no disk write)
        image_buf = await asyncio.to_thread(_prepare_image_buffer, image_path, 256)

        # Step 3 — POST file to VK upload server
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field(
                "photo", image_buf,
                filename=fname.replace(".png", ".jpg"),
                content_type="image/jpeg",
            )
            async with session.post(upload_url, data=form) as resp:
                upload_result = await resp.json(content_type=None)

        logger.info("[%s] VK upload server response: %s", fname, upload_result)

        server = upload_result.get("server")
        photo  = upload_result.get("photo")
        hash_  = upload_result.get("hash")

        if not photo or photo == "[]":
            logger.error("[%s] VK rejected the file — 'photo' field empty: %s", fname, upload_result)
            return None

        # Step 4 — save photo via VK API
        saved = await api.photos.save_messages_photo(
            photo=photo,
            server=server,
            hash=hash_,
        )
        logger.info("[%s] photos.saveMessagesPhoto response: %s", fname, saved)

        if not saved:
            logger.error("[%s] saveMessagesPhoto returned empty list", fname)
            return None

        photo_obj = saved[0]
        owner_id  = photo_obj.owner_id
        media_id  = photo_obj.id

        if not media_id:
            logger.error("[%s] media_id is 0 or missing in saved photo object: %s", fname, photo_obj)
            return None

        attachment = f"photo{owner_id}_{media_id}"
        logger.info("[%s] ✅ attachment string: %s", fname, attachment)
        return attachment

    except Exception as e:
        logger.error("[%s] Upload failed: %s", fname, e, exc_info=True)
        return None


async def upload_images_to_vk(bot, static_dir: str):
    """Upload all robot images to VK and cache attachment strings.

    Uploads are done sequentially with a 3-second gap to avoid VK rate limits.
    Each image gets up to 3 attempts before giving up.
    """
    global _attachments

    for key, fname in ROBOT_IMAGES.items():
        fpath = os.path.join(static_dir, fname)
        if not os.path.exists(fpath):
            logger.warning("Image file not found: %s", fpath)
            continue

        attachment = None
        for attempt in range(1, 4):
            if attempt > 1:
                wait = attempt * 3
                logger.info("[%s] Retry %d/%d — waiting %ds...", fname, attempt, 3, wait)
                await asyncio.sleep(wait)
            attachment = await _upload_single_image(bot, fpath)
            if attachment:
                break
            logger.warning("[%s] Upload attempt %d failed", fname, attempt)

        if attachment:
            _attachments[key] = attachment
            logger.info("Cached [%s] → %s", key, attachment)
        else:
            logger.warning("Could not upload [%s] after 3 attempts — messages will be text-only", key)

        # Pause between images to avoid VK rate limiting
        await asyncio.sleep(3)


def get_attachment(key: str) -> Optional[str]:
    return _attachments.get(key)
