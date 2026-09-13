"""统一日志模块：格式 [时间] [级别] [模块] 消息，10MB 轮转保留 5 份。

所有 tools/ 脚本与 src/ 模块共用。用法（在脚本开头）：

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.common.logger import setup_logger
    setup_logger("脚本名")   # 之后直接调用 logging.info(...) 等
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ===== 日志开关（规则：代码开头必须设） =====
LOG_ENABLED = True
LOG_LEVEL = logging.INFO

# 项目根 = src/common/ 上溯两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logger(name: str = "app") -> None:
    """配置 root logger：控制台 + 文件轮转（10MB × 5 份），重复调用幂等。"""
    if not LOG_ENABLED:
        logging.disable(logging.CRITICAL)
        return
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers.clear()

    fmt = "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(console)

    file_h = RotatingFileHandler(
        LOGS_DIR / f"{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_h.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(file_h)
