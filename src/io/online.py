"""网页视频链接 → 可拉流的直链（yt-dlp 解析）。

为什么需要这个模块
------------------
用户手里的往往不是流地址，而是**网页链接**（微博 / 抖音 / B站 / YouTube …）。
OpenCV/FFmpeg 打开网页拿到的是 HTML，不是视频——实测 `cv2.VideoCapture("https://weibo.com/tv/show/…")`
能"打开成功"但永远读不到帧（看门狗 30s 后判失败）。所以网页链接必须先解析出**媒体直链**再交给拉流层。

设计取舍
--------
- **只解析、不下载**：实测微博 CDN 直链 cv2 可直接拉流（首帧 0.2s）；下载落盘会引入分钟级等待与磁盘清理问题。
- **直链跳过解析**：`.mp4` / `.m3u8` 等本身就是媒体地址，直接拉流更快，也避免 yt-dlp 对它失败。
- **CDN 鉴权头透传**：yt-dlp 返回的 `http_headers`（UA / Referer / Cookie）转成
  `OPENCV_FFMPEG_CAPTURE_OPTIONS` 环境变量——有些站点（如 B站）的 CDN 校验 Referer/UA，
  不带头会 403。该变量在 `cv2.VideoCapture` 构造时读取。
- **yt-dlp 缺失时明确报错**：而不是抛 ImportError 到线程里变成一坨看不懂的堆栈。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

# 视为「媒体直链」的后缀——这些地址不需要（也不该）走 yt-dlp 解析
DIRECT_MEDIA_SUFFIXES = (".mp4", ".m3u8", ".flv", ".mkv", ".webm", ".ts", ".mov", ".avi")

# 透传给 FFmpeg http 协议的头（其余头没有对应的 capture 选项，透传只会被忽略）
_FFMPEG_HEADER_KEYS = {
    "user-agent": "user_agent",
    "referer": "referer",
    "cookie": "cookie",
}


@dataclass
class OnlineMedia:
    """一条解析结果：可直接交给 OpenCV 的拉流地址 + 需要的请求头。"""

    play_url: str
    headers: dict[str, str] = field(default_factory=dict)
    title: str = ""
    extractor: str = ""          # 来源站点名（Weibo / BiliBili / YouTube …）


def is_webpage_link(source) -> bool:
    """是否网页视频链接：http(s) 开头、且路径不是媒体直链后缀。"""
    text = str(source).strip().lower()
    if not text.startswith(("http://", "https://")):
        return False
    path = text.split("?", 1)[0].split("#", 1)[0]
    return not path.endswith(DIRECT_MEDIA_SUFFIXES)


def ffmpeg_capture_options(headers: dict[str, str]) -> str:
    """把 HTTP 头转成 `OPENCV_FFMPEG_CAPTURE_OPTIONS` 的值（`key;value|key;value`）。

    值里的 `;` 和 `|` 是该格式的分隔符，必须清掉，否则解析错位。
    """
    parts: list[str] = []
    for key, value in (headers or {}).items():
        name = _FFMPEG_HEADER_KEYS.get(key.lower())
        if not name:
            continue
        clean = str(value).replace(";", " ").replace("|", " ").strip()
        if clean:
            parts.append(f"{name};{clean}")
    return "|".join(parts)


def height_cap_for(max_side: int) -> int:
    """推理降采样宽度 → 请求的视频高度档。拉 1080p 再缩到 320px 纯属浪费带宽与解码。"""
    if max_side >= 960:
        return 1080
    if max_side >= 640:
        return 720
    return 480


# ---------------------------------------------------------------------------
# 站点专用适配器：yt-dlp 不覆盖、或被风控拦住的站点，在这里逐个补
# （好看视频实测：yt-dlp 报 Unsupported URL，老 API 已 404；页面是 JS 壳。
#   可用通道 = 先访问页面拿 BAIDUID Cookie，再 GET /v?pd=…&vid=…&_format=json）
# ---------------------------------------------------------------------------

def _resolve_haokan(url: str, max_height: int, timeout_s: float) -> OnlineMedia:
    """好看视频（haokan.baidu.com）专用解析。实测 2026-09：errno=0，cv2 可直拉。"""
    import requests

    vid = (parse_qs(urlparse(url).query).get("vid") or [""])[0]
    if not vid:
        raise RuntimeError("好看视频链接里没有 vid 参数")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "Chrome/146.0 Safari/537.36",
        "Referer": "https://haokan.baidu.com/",
    }
    session = requests.Session()
    session.headers.update(headers)
    try:
        session.get(f"https://haokan.baidu.com/v?pd=wisenatural&vid={vid}",
                    timeout=timeout_s)                      # 先拿 Cookie（BAIDUID 必需）
        r = session.get(f"https://haokan.baidu.com/v?pd=wisenatural&vid={vid}&_format=json",
                        timeout=timeout_s)
        data = r.json()
    except Exception as exc:
        raise RuntimeError(f"好看视频页面请求失败: {exc}") from exc
    if data.get("errno") != 0:
        raise RuntimeError(f"好看视频接口返回 errno={data.get('errno')}（视频可能已删除 / 需登录）")
    meta = (((data.get("data") or {}).get("apiData")) or {}).get("curVideoMeta") or {}
    play_url = meta.get("playurl") or ""
    if not play_url:
        raise RuntimeError("好看视频接口未返回播放地址")
    return OnlineMedia(play_url=play_url, headers=headers,
                       title=str(meta.get("title") or ""), extractor="HaoKan")


# 命中这些域名时优先走适配器；适配器失败再退回 yt-dlp（保持通用兜底）
SITE_ADAPTERS = {
    "haokan.baidu.com": _resolve_haokan,
}


def _host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def download_online(media: OnlineMedia, dest, max_bytes: int = 200 * 1024 * 1024,
                    timeout_s: float = 120.0):
    """把解析出的直链下载到本地文件（带鉴权头、限大小），返回 Path。

    为什么需要下载兜底：部分 CDN（实测 B站 upos）对 FFmpeg 携带的 Referer/UA 头
    不认账（OpenCV 5 的 FFmpeg 透传头无效，实测仍 403），但 requests 带头能通。
    这类站点只能"先下载再播放"——VOD 场景下载完成前无法开播，属如实取舍。
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(media.play_url, headers=media.headers or None,
                          stream=True, timeout=(10, 30)) as r:
            r.raise_for_status()
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    done += len(chunk)
                    if done > max_bytes:
                        raise RuntimeError(
                            f"视频超过 {max_bytes // 1048576} MB 下载上限，请换更短视频")
                    fh.write(chunk)
        tmp.replace(dest)
    except RuntimeError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"视频下载失败: {exc}") from exc
    log.info("[Online] 下载完成: %s (%.1f MB) -> %s", media.title[:40],
             dest.stat().st_size / 1048576, dest.name)
    return dest


def resolve_online(url: str, max_height: int = 720, timeout_s: float = 25.0) -> OnlineMedia:
    """解析网页视频链接为直链。失败抛 RuntimeError（message 面向用户可读）。

    顺序：站点专用适配器（对 yt-dlp 不支持的站点精准补漏）→ yt-dlp 通用解析。
    """
    adapter = SITE_ADAPTERS.get(_host_of(url))
    if adapter is not None:
        try:
            return adapter(url, max_height, timeout_s)
        except RuntimeError:
            raise
        except Exception as exc:  # 适配器自身的意外错误 → 退回 yt-dlp 再试一次
            log.warning("[Online] %s 适配器异常，退回 yt-dlp: %s", _host_of(url), exc)

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "服务端缺少 yt-dlp（pip install yt-dlp），暂不支持该网页视频链接") from exc

    # mp4 优先（OpenCV 拉流兼容性最好）；B站/YouTube 等是 DASH 分离流（无合成 mp4），
    # `bv*` 允许选中"纯视频"流——车牌识别用不到音频，纯视频反而省带宽。
    fmt = (f"bv*[ext=mp4][height<={max_height}]/bv*[height<={max_height}]"
           f"/best[ext=mp4][height<={max_height}]/best[height<={max_height}]/best")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": fmt,
        "socket_timeout": int(max(5, timeout_s)),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp 的 DownloadError 类型不稳定，统一按 RuntimeError 出
        raise RuntimeError(f"无法解析网页视频链接（可能需要登录 / 已删除 / 地区限制）: {exc}") from exc

    if info is None:
        raise RuntimeError("网页视频链接解析结果为空")
    if "entries" in info:                      # 命中了播放列表：取第一个视频
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise RuntimeError("播放列表里没有可用的视频")
        info = entries[0]

    play_url = info.get("url") or ""
    if not play_url:
        # 某些提取器把直链放在 formats 里而不在顶层——自己挑一个最匹配的
        best = None
        for f in info.get("formats") or []:
            if not f.get("url") or f.get("vcodec") == "none":
                continue
            h = f.get("height") or 0
            if best is None or (h and abs(h - max_height) < abs((best.get("height") or 0) - max_height)):
                best = f
        if best is None:
            raise RuntimeError("解析结果里没有可拉流的视频直链")
        play_url = best["url"]
        info.setdefault("http_headers", best.get("http_headers") or {})

    media = OnlineMedia(
        play_url=play_url,
        headers=dict(info.get("http_headers") or {}),
        title=str(info.get("title") or ""),
        extractor=str(info.get("extractor_key") or ""),
    )
    log.info("[Online] 解析成功: %s → %s (%s, %s)", url[:80], media.extractor,
             media.title[:40], f"{len(media.headers)} 个头")
    return media
