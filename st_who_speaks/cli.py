from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from st_who_speaks.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = list(sys.argv[1:] if argv is None else argv)
    app_path = Path(__file__).resolve().with_name("streamlit_app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *args]
    logger.info("launching streamlit", command=command)
    raise SystemExit(subprocess.run(command, check=False).returncode)
