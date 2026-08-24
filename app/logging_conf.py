import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # keep third-party libs from flooding stdout with debug/info noise
    for noisy in ("uvicorn.access", "httpx", "transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
