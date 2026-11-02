"""数据集验证子命令：``python -m engine.val --weights best.pt --data data.yaml``。

用于在测试页跑一遍验证集，拿到 mAP / 每类指标，与训练页的曲线相互印证。
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from . import report
from .metric_utils import per_class_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine.val",
        description="在数据集上评估权重（mAP 等指标通过 JSON Lines 上报）",
    )
    parser.add_argument("--weights", required=True, help="权重文件（.pt）")
    parser.add_argument("--data", required=True, help="数据集 data.yaml 路径")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="评估划分")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU 阈值")
    parser.add_argument("--device", default="", help="评估设备，留空自动")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False, help="保存评估图表")
    return parser


def _resolve(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _metric(values: list[float], index: int) -> float | None:
    try:
        return round(float(values[index]), 6)
    except (IndexError, TypeError, ValueError):
        return None


def run(args: argparse.Namespace) -> int:
    report.quiet()
    from ultralytics import YOLO

    weights = _resolve(args.weights)
    if not weights.is_file():
        report.fail(f"权重文件不存在：{weights}")
        return 2
    data_path = _resolve(args.data)
    if not data_path.is_file():
        report.fail(f"数据配置文件不存在：{data_path}")
        return 2

    report.log(f"评估 {weights.name} 于 {data_path} 的 {args.split} 划分")
    report.progress("val", 0, 1)

    progress_state = {"batches": 0, "done": 0}

    model = YOLO(str(weights))

    def on_val_batch_start(validator: Any) -> None:
        loader = getattr(validator, "dataloader", None)
        progress_state["batches"] = len(loader) if loader is not None else 0

    def on_val_batch_end(_validator: Any) -> None:
        total = progress_state["batches"]
        if not total:
            return
        progress_state["done"] += 1
        report.progress("val", progress_state["done"], total)

    model.add_callback("on_val_batch_start", on_val_batch_start)
    model.add_callback("on_val_batch_end", on_val_batch_end)

    try:
        metrics = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device or None,
            plots=args.plots,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report.fail(f"评估失败：{exc}")
        return 1

    report.progress("val", 1, 1)

    box = getattr(metrics, "box", None)
    values: dict[str, Any] = {}
    if box is not None:
        means = box.mean_results()
        for key, index in (("precision", 0), ("recall", 1), ("mAP50", 2), ("mAP50-95", 3)):
            value = _metric(means, index)
            if value is not None:
                values[key] = value

        speed = getattr(metrics, "speed", None) or {}
        for key, value in speed.items():
            try:
                values[f"speed_{key}_ms"] = round(float(value), 3)
            except (TypeError, ValueError):
                continue

        per_class = per_class_payload(box, getattr(metrics, "names", None))
        if per_class:
            values["per_class"] = per_class

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            try:
                values[f"metric/{key}"] = round(float(value), 6)
            except (TypeError, ValueError):
                continue

    if values:
        report.metrics(1, 1, values)

    highlighted = [f"{key} {values[key]:.4f}" for key in ("mAP50-95", "mAP50", "precision", "recall") if key in values]
    summary = f"评估完成（{args.split} 划分）"
    if highlighted:
        summary += "：" + "，".join(highlighted)
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        report.artifact("val_dir", Path(save_dir).resolve())
    report.done(summary, metrics=values)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
