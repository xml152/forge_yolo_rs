"""训练子命令：``python -m engine.train --data ... --epochs ...``。

进程 stdout 只输出协议事件，训练进度/指标/产物全部以 JSON Lines 上报，
Rust 侧解析后直接驱动进度条与指标表。
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
        prog="engine.train",
        description="训练 YOLO 模型（进度通过 JSON Lines 上报给 Rust 前端）",
    )
    parser.add_argument("--weights", default="weights/yolo26n.pt", help="初始化权重（.pt）")
    parser.add_argument("--data", required=True, help="数据集 data.yaml 路径")
    parser.add_argument("--epochs", type=int, default=100, help="总训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批大小，-1 为自动")
    parser.add_argument("--workers", type=int, default=8, help="数据加载进程数")
    parser.add_argument("--device", default="", help="训练设备，如 0 / 0,1 / cpu，留空自动")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值，0 表示关闭")
    parser.add_argument("--optimizer", default="auto", help="优化器：auto/SGD/Adam/AdamW")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--project", default="runs/detect", help="输出根目录")
    parser.add_argument("--name", default="train", help="本次训练子目录名")
    parser.add_argument(
        "--cache",
        default="none",
        choices=["none", "ram", "disk"],
        help="图像缓存方式",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="混合精度")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False, help="保存训练曲线图")
    parser.add_argument("--resume", action="store_true", help="从上次中断处续训")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    return parser


def _resolve(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _trainer_metric_object(trainer: Any) -> Any:
    """拿到本轮验证的 DetMetrics 对象（逐类指标要从这里取）。"""
    validator = getattr(trainer, "validator", None)
    return getattr(validator, "metrics", None) if validator is not None else None


def _metric_values(trainer: Any) -> dict[str, Any]:
    """汇总当前轮的 loss / 验证指标 / 学习率，外加逐类别 mAP。"""
    merged: dict[str, Any] = {}
    try:
        loss = trainer.label_loss_items(trainer.tloss, prefix="train")
    except Exception:
        loss = {}
    for part in (loss, getattr(trainer, "metrics", None), getattr(trainer, "lr", None)):
        if isinstance(part, dict):
            merged.update(part)

    values: dict[str, Any] = {}
    for key, value in merged.items():
        try:
            values[str(key)] = round(float(value), 6)
        except (TypeError, ValueError):
            continue

    det_metrics = _trainer_metric_object(trainer)
    box = getattr(det_metrics, "box", None) if det_metrics is not None else None
    if box is not None:
        per_class = per_class_payload(box, getattr(det_metrics, "names", None))
        if per_class:
            values["per_class"] = per_class
    return values


def run(args: argparse.Namespace) -> int:
    report.quiet()
    from ultralytics import YOLO

    data_path = _resolve(args.data)
    if not data_path.is_file():
        report.fail(f"数据配置文件不存在：{data_path}")
        return 2

    weights_path = _resolve(args.weights)
    if not weights_path.is_file():
        report.log(f"权重文件不存在，改为从零训练：{weights_path}", level="warn")
        weights_path = Path("yolo11n.yaml")

    model = YOLO(str(weights_path))

    counters = {"batches_total": 0, "batches_done": 0}
    final: dict[str, Any] = {}

    def on_pretrain_routine_end(trainer: Any) -> None:
        report.log(f"开始训练：{trainer.epochs} 轮，设备 {getattr(trainer, 'device', '?')}，数据 {data_path}")

    def on_train_epoch_start(trainer: Any) -> None:
        loader = getattr(trainer, "train_loader", None)
        counters["batches_total"] = len(loader) if loader is not None else 0
        counters["batches_done"] = 0
        # 轮级事件带 epoch/epochs，前端据此显示"第 N/M 轮"。
        report.progress("epoch", trainer.epoch, trainer.epochs, epoch=trainer.epoch + 1, epochs=trainer.epochs)

    def on_train_batch_end(_trainer: Any) -> None:
        total = counters["batches_total"]
        if not total:
            return
        counters["batches_done"] += 1
        report.progress(
            "batch",
            min(counters["batches_done"], total),
            total,
            percent=min(counters["batches_done"], total) / total,
        )

    def on_fit_epoch_end(trainer: Any) -> None:
        # 只有到这里 self.metrics 才是本轮的验证结果（on_train_epoch_end 时还是上一轮的）。
        # final_eval 会把 epoch 临时置为 epochs 再触发一次本事件，那次是"最佳权重复评"。
        values = _metric_values(trainer)
        if not values:
            return
        if trainer.epoch >= trainer.epochs:
            report.metrics(trainer.epochs, trainer.epochs, values, final=True)
            return
        final.clear()
        final.update(values)
        report.metrics(trainer.epoch + 1, trainer.epochs, values)

    def on_train_end(trainer: Any) -> None:
        for kind, path in (
            ("best", getattr(trainer, "best", None)),
            ("last", getattr(trainer, "last", None)),
        ):
            if path is not None and Path(path).is_file():
                report.artifact(kind, Path(path).resolve())
        report.log(f"训练结束，结果目录：{Path(getattr(trainer, 'save_dir', args.project)).resolve()}")

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    try:
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device or None,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            project=str(_resolve(args.project)),
            name=args.name,
            cache=False if args.cache == "none" else args.cache,
            amp=args.amp,
            plots=args.plots,
            resume=args.resume,
            seed=args.seed,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - 任何失败都要原样报回前端
        traceback.print_exc()
        report.fail(f"训练失败：{exc}")
        return 1

    trainer = getattr(model, "trainer", None)
    best = Path(getattr(trainer, "best", "")) if trainer is not None else None
    summary = "训练完成"
    if best is not None and best.is_file():
        summary = f"训练完成，最佳权重：{best.resolve()}"
    report.done(summary, metrics=final)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())