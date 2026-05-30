import logging
import aiohttp
from config import JDOODLE_CLIENT_ID, JDOODLE_CLIENT_SECRET, JDOODLE_PLAN, JDOODLE_FREE_LIMIT, JDOODLE_ENABLED

logger = logging.getLogger(__name__)

JDOODLE_API_URL = "https://api.jdoodle.com/v1/execute"
JDOODLE_CREDIT_URL = "https://api.jdoodle.com/v1/credit-spent"

JDOODLE_JAVA_VERSION = "4"  # Java 17

_WRAPPER_TEMPLATE = (
    "public class Main {{\n"
    "    public static void main(String[] args) {{\n"
    "        {body}\n"
    "    }}\n"
    "}}"
)

_METHOD_WRAPPER_TEMPLATE = (
    "public class Main {{\n"
    "{body}\n"
    "}}"
)


def _prepare_code(code: str) -> str:
    """Ensure code is a valid JDoodle submission.

    Rules:
    - If code already has `public class` → use as-is
    - If code has `class` but not `public class` → prepend `public `
    - If code looks like method(s) (contains `static`) but no class → wrap in class
    - Otherwise → wrap in public class Main with main method
    """
    stripped = code.strip()

    # Already a complete public class
    if "public class" in stripped:
        return stripped

    # Has a class declaration but missing public
    if "class " in stripped and "{" in stripped:
        return "public " + stripped

    # Contains method definitions (has `static` + `{`) but no class wrapper
    if "static" in stripped and "void" in stripped and "{" in stripped:
        return _METHOD_WRAPPER_TEMPLATE.format(body=stripped)

    # Bare statements — wrap in main method
    # Indent each line
    indented = "\n        ".join(stripped.splitlines())
    return _WRAPPER_TEMPLATE.format(body=indented)


async def execute_java_code(code: str) -> dict:
    """Execute Java code via JDoodle API.

    Returns dict:
        output  — stdout/stderr text
        error   — True if compilation/runtime error occurred
        message — human-readable status line
        limit_exceeded — True if daily limit reached
        service_error  — True if JDoodle is unreachable / misconfigured
    """
    if not JDOODLE_ENABLED:
        logger.warning("JDoodle: credentials not configured")
        return {
            "output": "",
            "error": False,
            "message": "not_configured",
            "limit_exceeded": False,
            "service_error": True,
        }

    prepared = _prepare_code(code)
    payload = {
        "clientId": JDOODLE_CLIENT_ID,
        "clientSecret": JDOODLE_CLIENT_SECRET,
        "script": prepared,
        "language": "java",
        "versionIndex": JDOODLE_JAVA_VERSION,
    }

    key_preview = JDOODLE_CLIENT_ID[:6] + "..."
    logger.info("JDoodle request → clientId: %s | plan: %s | original_len: %d | prepared_len: %d",
                key_preview, JDOODLE_PLAN, len(code), len(prepared))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                JDOODLE_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                status = resp.status
                data = await resp.json(content_type=None)

                logger.info("JDoodle response → HTTP %d | statusCode: %s | memory: %s | cpuTime: %s",
                            status, data.get("statusCode"), data.get("memory"), data.get("cpuTime"))

                if status != 200:
                    logger.error("JDoodle HTTP error %d: %s", status, str(data)[:300])
                    return {
                        "output": "",
                        "error": True,
                        "message": f"HTTP {status}",
                        "limit_exceeded": False,
                        "service_error": True,
                    }

                jd_status = data.get("statusCode", 200)
                output = (data.get("output") or "").strip()

                # JDoodle statusCode 429 = daily credit limit reached server-side
                if jd_status == 429:
                    logger.warning("JDoodle: server-side daily limit reached")
                    return {
                        "output": output,
                        "error": False,
                        "timeout": False,
                        "message": "limit_exceeded",
                        "limit_exceeded": True,
                        "service_error": False,
                    }

                # Timeout detection: JDoodle reports TLE in the output text
                timeout_markers = (
                    "Time Limit Exceeded",
                    "time limit exceeded",
                    "TimeLimitExceeded",
                )
                has_timeout = any(m in output for m in timeout_markers)

                # Compilation / runtime error detection: JDoodle puts error info in output
                error_markers = (
                    "error:", "Error:", "Exception", "error\n",
                    "compilation error", "Main.java:", ".java:",
                    "No \"public class\"", "cannot find symbol",
                    "illegal start", "reached end of file",
                )
                has_error = (
                    jd_status not in (200, None) or
                    any(marker in output for marker in error_markers)
                )

                logger.info(
                    "JDoodle result → has_error: %s | has_timeout: %s | output_len: %d",
                    has_error, has_timeout, len(output),
                )

                return {
                    "output": output,
                    "error": has_error,
                    "timeout": has_timeout,
                    "message": "ok",
                    "limit_exceeded": False,
                    "service_error": False,
                }

    except aiohttp.ClientConnectorError as e:
        logger.error("JDoodle: connection error: %s", e)
    except aiohttp.ServerTimeoutError as e:
        # Our HTTP client timed out waiting for JDoodle — treat as program timeout
        logger.error("JDoodle: HTTP timeout waiting for response: %s", e)
        return {
            "output": "",
            "error": True,
            "timeout": True,
            "message": "http_timeout",
            "limit_exceeded": False,
            "service_error": False,
        }
    except aiohttp.ClientError as e:
        logger.error("JDoodle: client error: %s", e)
    except Exception as e:
        logger.error("JDoodle: unexpected error: %s", e, exc_info=True)

    return {
        "output": "",
        "error": True,
        "timeout": False,
        "message": "network_error",
        "limit_exceeded": False,
        "service_error": True,
    }
