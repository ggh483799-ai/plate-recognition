"""浏览器摄像头会话注册表：并发上限、空闲回收、产物目录。

与 `service/streams.py`（服务端拉流会话）的分工
--------------------------------------------
- `streams.py`：服务端去**拉**流（摄像头索引 / RTSP / 网页链接 / 本地视频）；
- 本模块：帧由**浏览器推**上来（`ClientCameraSession`），服务端只识别不回画面。

两者共用同一套并发闸门思路：CPU 是唯一瓶颈，超过上限**直接拒绝**，
而不是让几路会话互相抢核、把每一路都拖成幻灯片。

空闲回收的口径不同
----------------
拉流会话能感知"客户端断开"（MJPEG 长连接断掉）；浏览器推帧没有长连接，
所以判据是**多久没收到帧**——用户关掉标签页/拔网线后，超过 `idle_timeout_s`
没帧就自动收尾，把算力位让出来。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from service.jobs import _remove_tree
from service.streams import StreamManager
from src.pipeline.browser_camera import ClientCameraSession

log = logging.getLogger(__name__)

MAX_CAMERAS = 2
DEFAULT_IDLE_TIMEOUT_S = 45.0     # 多久没收到帧就认为用户走了
DEFAULT_TTL_S = 900.0             # 已结束会话的记录保留 15 分钟（产物可继续下载）

# 产物目录的对外地址前缀（与拉流会话的 /stream-media 分开挂载，互不混淆）
MEDIA_BASE_PREFIX = "/camera-media"


class TooManyCameras(RuntimeError):
    """并发路数超上限。"""


class CameraSessionManager:
    """浏览器摄像头会话注册表（线程安全：只保护注册表本身）。"""

    def __init__(
        self,
        root: str | Path,
        pipeline_factory: Callable[..., object],
        max_sessions: int = MAX_CAMERAS,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        ttl_s: float = DEFAULT_TTL_S,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._factory = pipeline_factory
        self.max_sessions = max(1, int(max_sessions))
        self.idle_timeout_s = float(idle_timeout_s)
        self.ttl_s = float(ttl_s)
        self._sessions: dict[str, ClientCameraSession] = {}
        self._lock = threading.Lock()

    # ---------------- 校验（阈值/尺寸口径与拉流会话保持一致） ----------------

    clamp_max_side = staticmethod(StreamManager.clamp_max_side)
    clamp_interval = staticmethod(StreamManager.clamp_interval)

    # ---------------- 生命周期 ----------------

    def create(
        self,
        session_id: str,
        *,
        name: str = "浏览器摄像头",
        interval_ms: int = 200,
        level: str | None = None,
        max_side: int = 640,
        min_det_score: float | None = None,
        min_rec_score: float | None = None,
        show_objects: bool = True,
    ) -> ClientCameraSession:
        """启动一路浏览器摄像头会话。**会阻塞数秒**（加载模型），调用方应放线程池。"""
        self.expire_idle()
        with self._lock:
            active = [s for s in self._sessions.values() if not s.is_finished]
            if len(active) >= self.max_sessions:
                raise TooManyCameras(
                    f"已有 {len(active)} 路摄像头会话在跑（上限 {self.max_sessions}），请先停止其中一路"
                )

        # 每路会话独立模型实例：跨线程共享不安全（与拉流会话同样的理由）
        pipeline = self._factory(level, min_det_score, min_rec_score)
        if pipeline is None:
            raise RuntimeError("实时流水线不可用（模型未就绪）")

        session = ClientCameraSession(
            session_id,
            pipeline,
            name=name,
            out_dir=self.root / session_id,
            interval_ms=self.clamp_interval(interval_ms),
            max_side=self.clamp_max_side(max_side),
            idle_timeout_s=self.idle_timeout_s,
            show_objects=show_objects,
            media_base_prefix=MEDIA_BASE_PREFIX,
        )
        with self._lock:
            self._sessions[session_id] = session
        log.info("[Camera] 会话已创建: %s interval=%dms size=%d level=%s",
                 session_id, session.interval_s * 1000, session.max_side, level or "默认")
        return session

    def get(self, session_id: str) -> ClientCameraSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def snapshot(self, session_id: str) -> dict | None:
        session = self.get(session_id)
        return session.snapshot() if session is not None else None

    def ingest(self, session_id: str, data: bytes) -> dict | None:
        """喂一帧给会话；会话不存在返回 None（调用方回 404）。"""
        session = self.get(session_id)
        if session is None:
            return None
        return session.ingest(data)

    def stop(self, session_id: str) -> bool:
        session = self.get(session_id)
        if session is None:
            return False
        session.stop()
        return True

    def remove(self, session_id: str) -> dict | None:
        """停止会话并删除产物目录（会话不存在返回 None）。"""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        session.stop()
        purged = _remove_tree(session.out_dir) if session.out_dir is not None else True
        if not purged:
            log.warning("[Camera] %s 已移出列表，但产物目录仍在磁盘: %s", session_id, session.out_dir)
        return {"session_id": session_id, "purged": purged}

    def expire_idle(self) -> int:
        """把"长时间没有新帧"的会话收尾（用户关掉页面的兜底回收）。"""
        with self._lock:
            stale = [s for s in self._sessions.values()
                     if not s.is_finished
                     and s.idle_seconds() > s.idle_timeout_s]
        for session in stale:
            log.info("[Camera] %s 超过 %.0fs 没有收到帧，自动收尾",
                     session.session_id, session.idle_timeout_s)
            session.stop()
        return len(stale)

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
            log.info("[Camera] 清理过期会话 %d 个%s", len(stale),
                     f"（{unpurged} 个产物因环境限制保留在磁盘）" if unpurged else "")
        return len(stale)

    def count(self) -> dict:
        self.expire_idle()
        with self._lock:
            items = list(self._sessions.values())
        return {
            "total": len(items),
            "active": sum(1 for s in items if not s.is_finished),
            "max": self.max_sessions,
        }

    def stop_all(self, timeout: float = 2.0) -> None:
        """服务关停时收尾（lifespan 用）。"""
        with self._lock:
            items = list(self._sessions.values())
        for s in items:
            if not s.is_finished:
                s.stop()
