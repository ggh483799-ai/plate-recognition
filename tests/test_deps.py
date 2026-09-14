"""依赖清单守卫：代码直接 import 的第三方包，必须在部署侧声明。

背景（2026-09-14 真实事故）
--------------------------
`yt-dlp` 只在本机 venv 装过、没写进 `requirements.txt`，本地 151 个测试全绿、
CI 也全绿——但容器里没有它，线上「网页视频链接」功能直接报
「服务端缺少 yt-dlp，暂不支持该网页视频链接」。

本地环境比部署环境"多装了东西"，是测试测不出来的那类漂移。
这组用例把「装过但没声明」在本地就拦住：拿代码里真实出现的第三方 import，
去对 requirements.txt（或 Dockerfile 的 --no-deps 安装）做核对。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
DOCKERFILE = ROOT / "docker" / "Dockerfile"

# import 名 → PyPI 发行名（只有不一致的才需要映射）
IMPORT_TO_DIST = {
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "yt_dlp": "yt-dlp",
}

# src/ + service/ 中真实出现的第三方顶层 import
REQUIRED_IMPORTS = [
    "PIL", "cv2", "fastapi", "numpy", "onnxruntime", "pydantic",
    "requests", "torch", "ultralytics", "yaml", "yt_dlp",
]

# 故意不进 requirements.txt 的包：必须由 Dockerfile 显式安装（并写明原因）
DOCKERFILE_ONLY = {
    "hyperlpr3": "声明依赖 opencv-python 与 fastapi==0.92，直接用 requirements 装会"
                 "覆盖 headless opencv、降级 fastapi，故 Dockerfile 用 --no-deps 单独装",
}


def _declared_in_requirements() -> set[str]:
    """requirements.txt 中生效（未被注释）的发行名集合，小写。"""
    names: set[str] = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[; ]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def _installed_in_dockerfile() -> set[str]:
    """Dockerfile 里 pip install 过的包名（含 --no-deps 单独装的那些）。"""
    text = DOCKERFILE.read_text(encoding="utf-8")
    names: set[str] = set()
    for line in text.splitlines():
        if "pip install" not in line:
            continue
        # 去掉 pip install 的开关（--no-cache-dir / --no-deps / -r 等）
        args = line.split("pip install", 1)[1]
        args = args.split("-r ", 1)[0]          # -r requirements.txt 之后不算具体包
        for tok in args.split():
            if tok.startswith("-") or "/" in tok or "\\" in tok:
                continue
            names.add(tok.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0]
                      .split("=", 1)[0].strip().lower())
    return {n for n in names if n}


def test_requirements_and_dockerfile_exist():
    assert REQUIREMENTS.is_file(), "缺少 requirements.txt"
    assert DOCKERFILE.is_file(), "缺少 docker/Dockerfile"


def test_direct_imports_are_declared():
    """代码直接 import 的包，必须能在部署侧找到（requirements 或 Dockerfile）。"""
    declared = _declared_in_requirements() | _installed_in_dockerfile()
    missing = []
    for imp in REQUIRED_IMPORTS:
        dist = IMPORT_TO_DIST.get(imp, imp).lower()
        if dist not in declared:
            missing.append(f"{imp} → {dist}")
    assert not missing, (
        "以下第三方包被代码直接 import，但既不在 requirements.txt，"
        f"也不在 Dockerfile 安装：{missing}"
    )


def test_webpage_link_runtime_deps_declared():
    """网页链接解析链路的运行时依赖单独立一条，防止再次「本机装了、线上没有」。"""
    declared = _declared_in_requirements()
    assert "yt-dlp" in declared, "yt-dlp 未声明——线上网页视频链接会直接不可用"
    assert "requests" in declared, "requests 未声明（代码直接 import 使用）"


def test_dockerfile_only_packages_are_really_in_dockerfile():
    """刻意不写进 requirements 的包，必须真的由 Dockerfile 装上（否则线上缺件）。"""
    installed = _installed_in_dockerfile()
    for dist in DOCKERFILE_ONLY:
        assert dist in installed, f"{dist} 既不在 requirements，也没被 Dockerfile 安装"


def test_no_hard_requirement_left_commented_out():
    """被注释掉的依赖行不算声明——防止有人把必需项注释掉还以为它在。"""
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = raw.lstrip()
        if not stripped.startswith("#"):
            continue
        assert not re.match(r"#\s*(yt-dlp|requests|pillow)\s*[<>=]", stripped), (
            f"必需依赖被注释掉了：{stripped}"
        )
