import asyncio
import logging
import os
import struct
import zlib
from typing import Optional

logger = logging.getLogger(__name__)

# Cache of uploaded attachment strings
_attachments: dict[str, str] = {}

ROBOT_IMAGES = {
    "greeting": "robot_greeting.png",
    "thinking": "robot_thinking.png",
    "happy": "robot_happy.png",
    "sad": "robot_sad.png",
}


def _create_minimal_png(color_rgb: tuple[int, int, int] = (100, 149, 237), size: int = 64) -> bytes:
    """Create a minimal valid PNG image with given color."""
    def make_png(width, height, rgb):
        def crc(data):
            return struct.pack('>I', zlib.crc32(data) & 0xffffffff)

        def chunk(ctype, data):
            return struct.pack('>I', len(data)) + ctype + data + crc(ctype + data)

        # Signature
        sig = b'\x89PNG\r\n\x1a\n'
        # IHDR
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
        ihdr = chunk(b'IHDR', ihdr_data)
        # IDAT
        raw_data = b''
        for _ in range(height):
            raw_data += b'\x00' + bytes(rgb) * width
        compressed = zlib.compress(raw_data)
        idat = chunk(b'IDAT', compressed)
        # IEND
        iend = chunk(b'IEND', b'')
        return sig + ihdr + idat + iend

    return make_png(size, size, color_rgb)


def ensure_static_images(static_dir: str):
    """Create placeholder PNG files if they don't exist."""
    os.makedirs(static_dir, exist_ok=True)
    colors = {
        "robot_greeting.png": (70, 130, 180),   # steel blue
        "robot_thinking.png": (255, 165, 0),     # orange
        "robot_happy.png": (50, 205, 50),        # green
        "robot_sad.png": (220, 20, 60),          # crimson
    }
    for fname, color in colors.items():
        fpath = os.path.join(static_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "wb") as f:
                f.write(_create_minimal_png(color))
            logger.info("Created placeholder: %s", fpath)


async def upload_images_to_vk(bot, static_dir: str):
    """Upload all robot images to VK and cache attachment strings."""
    global _attachments

    for key, fname in ROBOT_IMAGES.items():
        fpath = os.path.join(static_dir, fname)
        if not os.path.exists(fpath):
            logger.warning("Image not found: %s", fpath)
            continue
        try:
            attachment = await _upload_single_image(bot, fpath)
            if attachment:
                _attachments[key] = attachment
                logger.info("Uploaded %s → %s", fname, attachment)
            else:
                logger.warning("Failed to upload %s", fname)
        except Exception as e:
            logger.error("Error uploading %s: %s", fname, e)


def _resize_image(image_path: str, max_size: int = 512) -> "io.BytesIO":
    """Open image, resize so the longest side <= max_size, return PNG BytesIO buffer."""
    import io
    from PIL import Image
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    new_w, new_h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    logger.info("Resized %s: %dx%d → %dx%d", os.path.basename(image_path), orig_w, orig_h, new_w, new_h)
    return buf


async def _upload_single_image(bot, image_path: str) -> Optional[str]:
    """Upload a single image to VK messages (resized), return attachment string."""
    try:
        import aiohttp
        api = bot.api

        # Get upload server URL (peer_id=0 works for community bots)
        upload_server = await api.photos.get_messages_upload_server(peer_id=0)
        upload_url = upload_server.upload_url

        # Resize image to max 512px before uploading
        image_buf = await asyncio.to_thread(_resize_image, image_path, 512)

        # Upload resized image from buffer
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field(
                "photo", image_buf,
                filename=os.path.basename(image_path),
                content_type="image/png"
            )
            async with session.post(upload_url, data=form) as resp:
                result = await resp.json(content_type=None)

        if "photo" not in result:
            logger.error("VK upload response missing 'photo' key: %s", result)
            return None

        # Save photo
        saved = await api.photos.save_messages_photo(
            photo=result["photo"],
            server=result["server"],
            hash=result["hash"],
        )
        if saved:
            photo = saved[0]
            return f"photo{photo.owner_id}_{photo.id}"
    except Exception as e:
        logger.error("VK upload error for %s: %s", image_path, e)
    return None


def get_attachment(key: str) -> Optional[str]:
    return _attachments.get(key)
