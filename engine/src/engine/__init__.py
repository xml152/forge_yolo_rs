"""Forge YOLO 的 Python 引擎。

Rust 前端以子命令方式调用：``python -m engine <train|infer|val|export> ...``，
每个子命令都通过 :mod:`engine.report` 以 JSON Lines 汇报进度与结果。
"""

from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "train": "训练模型",
    "infer": "推理图片",
    "val": "评估权重",
    "export": "导出模型",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="engine", description="Forge YOLO Python 引擎")
        parser.add_argument("command", nargs="?", choices=sorted(COMMANDS), help="要执行的子命令")
        parser.parse_args(argv if argv else [])
        print("可用子命令：")
        for name, text in COMMANDS.items():
            print(f"  {name:<8}{text}")
        return 2 if not argv else 0

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"未知子命令：{command}（可用：{', '.join(COMMANDS)}）", file=sys.stderr)
        return 2

    if command == "train":
        from .train import main as entry
    elif command == "infer":
        from .infer import main as entry
    elif command == "val":
        from .val import main as entry
    else:
        from .export import main as entry

    return entry(rest)


__all__ = ["main"]
