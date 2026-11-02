"""Rust 与 Python 引擎之间的 JSON Lines 上报通道。

约定：进程的 stdout 只承载协议事件（每行一个 JSON 对象），ultralytics 的日志、
进度条以及任何第三方 ``print`` 全部重定向到 stderr，由 Rust 侧当作普通日志显示。
这样协议通道永远不会被库的杂散输出污染。

``sys.stdout`` 的替换发生在导入本模块时，因此各子命令必须在本模块之后
再导入 ultralytics。
"""

from __future__ import annotations

import json
import sys
from typing import Any

_PROTOCOL_STREAM = sys.stdout
sys.stdout = sys.stderr


def _fallback(value: Any) -> Any:
    """把张量等非 JSON 类型转成基本类型。"""
    for caster in (float, int, str):
        try:
            return caster(value)
        except (TypeError, ValueError):
            continue
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return repr(value)


def emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, default=_fallback)
    _PROTOCOL_STREAM.write(line + "\n")
    _PROTOCOL_STREAM.flush()


def log(message: str, level: str = "info") -> None:
    emit({"type": "log", "level": level, "message": str(message)})


def progress(stage: str, current: float, total: float, **extra: Any) -> None:
    payload = {"type": "progress", "stage": stage, "current": float(current), "total": float(total)}
    payload.update(extra)
    emit(payload)


def metrics(epoch: int, epochs: int, values: dict[str, Any], **extra: Any) -> None:
    payload = {"type": "metrics", "epoch": int(epoch), "epochs": int(epochs), "values": values}
    payload.update(extra)
    emit(payload)


def artifact(kind: str, path: Any, **extra: Any) -> None:
    payload = {"type": "artifact", "kind": str(kind), "path": str(path)}
    payload.update(extra)
    emit(payload)


def result(payload: dict[str, Any]) -> None:
    emit({"type": "result", "payload": payload})


def done(summary: str, **extra: Any) -> None:
    payload = {"type": "done", "summary": str(summary)}
    payload.update(extra)
    emit(payload)


def fail(message: str) -> None:
    emit({"type": "error", "message": str(message)})


def quiet() -> None:
    """关掉 ultralytics 自带的 tqdm 进度条（进度改由 progress 事件提供）。"""
    try:
        import ultralytics.utils

        ultralytics.utils.VERBOSE = False
    except Exception:
        pass
