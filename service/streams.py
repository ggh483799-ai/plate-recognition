"""实时流会话注册表：并发上限、生命周期、产物目录。

与 `service/jobs.py`（视频异步任务）的分工
----------------------------------------
- `jobs.py` 管**有终点**的任务：把一整段视频处理完，产出文件后结束；
- 本模块管**没有终点**的会话：摄像头 / RTSP / 边播边识别，直到用户停止、源结束，
  或长时间没有客户端消费（浏览器关了标签页）。

共同点：CPU 是唯一瓶颈，所以并发必须设硬上限（默认 2 路）。超了**直接拒绝**，
而不是让几路流互相抢核、把每一路都拖成幻灯片——那对谁都没价值。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from service.jobs import _remove_tree
from src.pipeline.stream import LiveStreamSession

log = logging.getLogger(__name__)

MAX_STREAMS = 2
DEFAULT_IDLE_TIMEOUT_S = 60.0     # 没人看就自动停
DEFAULT_TTL_S = 1800.0            # 会话记录保留 30 分钟（产物可继续下载）
MAX_SIDE_CHOICES = (320, 480, 640, 960, 1280)


class TooManyStreams(RuntimeError):
    """并发路数超上限。"""


class StreamManager:
    """会话注册表 + 并发闸门。线程安全（只保护注册表本身，会话内部各自加锁）。"""

    def __init__(
        self,
        root: str | Path,
        pipeline_factory: Callable[..., object],
        max_streams: int = MAX_STREAMS,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        ttl_s: float = DEFAULT_TTL_S,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._factory = pipeline_factory
        self.max_streams = max(1, int(max_streams))
        self.idle_timeout_s = float(idle_timeout_s)
        self.ttl_s = float(ttl_s)
        self._sessions: dict[str, LiveStreamSession] = {}
        self._lock = threading.Lock()

    # ---------------- 校验 ----------------

    @staticmethod
    def clamp_max_side(value: int) -> int:
        """把画面宽度收敛到允许档位（越宽越慢，是实时场景最有效的加速旋钮）。"""
        try:
            value = int(value or 0)
        except (TypeError, ValueError):
            value = 0
        return min(MAX_SIDE_CHOICES, key=lambda c: abs(c - value)) if value else 640

    @staticmethod
    def clamp_interval(value: int) -> int:
        """识别间隔（毫秒）：0 = 每帧都试（尽力）；上限 5s，免得框几秒不动。"""
        try:
            value = int(value or 0)
        except (TypeError, ValueError):
            value = 0
        return max(0, min(value, 5000))

    @staticmethod
    def clamp_quality(value: int) -> int:
        try:
            value = int(value or 0)
        except (TypeError, ValueError):
            value = 0
        return max(40, min(value or 75, 95))

    # ---------------- 生命周期 ----------------

    def create(
        self,
        session_id: str,
        source,
        *,
        out_dir: str | Path | None = None,
        name: str = "",
        interval_ms: int = 0,
        level: str | None = None,
        max_side: int = 640,
        jpeg_quality: int = 75,
        min_det_score: float | None = None,
        min_rec_score: float | None = None,
        show_objects: bool = True,
    ) -> LiveStreamSession:
        """构建流水线并启动一路会话。**会阻塞数秒**（加载模型），调用方应放到线程池里跑。

        `min_det_score` / `min_rec_score`：按会话覆盖识别阈值——玩具车、小车牌、远距离车牌
        经常低于默认 0.5/0.6，调低才有机会被识别（页面上的「灵敏度」旋钮）。
        """
        with self._lock:
            active = [s for s in self._sessions.values() if not s.is_finished]
            if len(active) >= self.max_streams:
                raise TooManyStreams(
                    f"已有 {len(active)} 路实时会话在跑（上限 {self.max_streams}），请先停止其中一路"
                )

        # 每路会话独立实例：跨线程共享模型不安全
        pipeline = self._factory(level, min_det_score, min_rec_score)
        if pipeline is None:
            raise RuntimeError("实时流水线不可用（模型未就绪）")

        session = LiveStreamSession(
            session_id,
            source,
            pipeline,
            name=name,
            out_dir=out_dir,
            interval_ms=self.clamp_interval(interval_ms),
            max_side=self.clamp_max_side(max_side),
            jpeg_quality=self.clamp_quality(jpeg_quality),
            idle_timeout_s=self.idle_timeout_s,
            show_objects=show_objects,
        )
        with self._lock:
            self._sessions[session_id] = session
        session.start()
        return session

    def get(self, session_id: str) -> LiveStreamSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def snapshot(self, session_id: str) -> dict | None:
        session = self.get(session_id)
        return session.snapshot() if session is not None else None

    def stop(self, session_id: str) -> bool:
        """停止会话但**保留记录**（事件列表与截图仍可查）。"""
        session = self.get(session_id)
        if session is None:
            return False
        session.stop()
        return True

    def remove(self, session_id: str) -> dict | None:
        """停止会话并删除产物目录。会话不存在返回 None。"""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.stop()
        out_dir = session.out_dir
        purged = _remove_tree(out_dir) if out_dir is not None else True
        if not purged:
            log.warning("[Stream] %s 已移出列表，但产物目录仍在磁盘: %s", session_id, out_dir)
        return {"session_id": session_id, "purged": purged}

    def purge(self) -> int:
        """清掉「已结束且超过 TTL」的会话记录与产物。"""
        now = time.time()
        with self._lock:
            stale = [s for s in self._sessions.values()
                     if s.is_finished and s.finished_at and (now - s.finished_at) > self.ttl_s]
            for s in stale:
                self._sessions.pop(s.session_id, None)
        unpurged = 0
        for s in stale:
            if s.out_dir is not None and not _remove_tree(s.out_dir):
                unpurged += 1
        if stale:
            log.info("[Stream] 清理过期会话 %d 个%s", len(stale),
                     f"（{unpurged} 个产物因环境限制保留在磁盘）" if unpurged else "")
        return len(stale)

    def count(self) -> dict:
        with self._lock:
            items = list(self._sessions.values())
        return {
            "total": len(items),
            "active": sum(1 for s in items if not s.is_finished),
            "max": self.max_streams,
        }

    def stop_all(self, timeout: float = 2.0) -> None:
        """服务关停时收尾（lifespan 用）。"""
        with self._lock:
            items = list(self._sessions.values())
        for s in items:
            if not s.is_finished:
                s.stop(timeout=timeout)
