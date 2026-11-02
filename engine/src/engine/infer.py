"""推理子命令：``python -m engine.infer --weights best.pt --source imgs/``。

对每张图片上报一条 ``result`` 事件（含整型像素坐标与归一化坐标），
Rust 侧既可以自己在图上画框，也可以直接用 ``--save`` 生成的标注图。
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

from . import report

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine.infer",
        description="用训练好的权重做推理（结果通过 JSON Lines 上报）",
    )
    parser.add_argument("--weights", required=True, help="权重文件（.pt）")
    parser.add_argument("--source", required=True, help="图片文件或目录")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--device", default="", help="推理设备，留空自动")
    parser.add_argument("--max-det", type=int, default=300, help="单张图最多保留的检测框数")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="保存带框的标注图")
    parser.add_argument("--project", default="runs/predict", help="输出根目录")
    parser.add_argument("--name", default="test", help="本次推理子目录名")
    parser.add_argument("--txt", action=argparse.BooleanOptionalAction, default=False, help="同时导出 YOLO txt")
    return parser


def _resolve(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _collect(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def _detections(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    height, width = result.orig_shape

    detections: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        class_id = int(classes[i])
        detections.append(
            {
                "class_id": class_id,
                "name": str(result.names.get(class_id, f"class_{class_id}")),
                "confidence": round(float(confs[i]), 6),
                "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "xywhn": [
                    round((x1 + x2) / 2 / width, 6),
                    round((y1 + y2) / 2 / height, 6),
                    round((x2 - x1) / width, 6),
                    round((y2 - y1) / height, 6),
                ],
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def run(args: argparse.Namespace) -> int:
    report.quiet()
    from ultralytics import YOLO

    weights = _resolve(args.weights)
    if not weights.is_file():
        report.fail(f"权重文件不存在：{weights}")
        return 2

    source = _resolve(args.source)
    images = _collect(source)
    if not images:
        report.fail(f"没有找到图片：{source}")
        return 2
    report.log(f"待推理图片 {len(images)} 张，权重 {weights}")

    try:
        model = YOLO(str(weights))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report.fail(f"加载权重失败：{exc}")
        return 1

    save_dir = _resolve(args.project) / args.name
    total_detections = 0
    elapsed: list[float] = []

    for index, image in enumerate(images, start=1):
        try:
            results = model.predict(
                source=str(image),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device or None,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            report.fail(f"推理失败：{image}：{exc}")
            return 1

        for result in results:
            detections = _detections(result)
            total_detections += len(detections)
            saved: str | None = None
            if args.save:
                save_dir.mkdir(parents=True, exist_ok=True)
                saved = str(Path(result.save(filename=str(save_dir / f"{image.stem}.jpg"))).resolve())
            if args.txt and detections:
                save_dir.mkdir(parents=True, exist_ok=True)
                result.save_txt(str(save_dir / f"{image.stem}.txt"), save_conf=True)

            speed = {k: float(v) for k, v in (result.speed or {}).items() if v is not None}
            inference_ms = speed.get("inference")
            if inference_ms:
                elapsed.append(inference_ms)

            report.result(
                {
                    "image": str(image),
                    "width": int(result.orig_shape[1]),
                    "height": int(result.orig_shape[0]),
                    "detections": detections,
                    "speed": speed,
                    "annotated": saved,
                }
            )

        report.progress("predict", index, len(images))

    summary = f"推理完成：{len(images)} 张图片，{total_detections} 个目标"
    extras: dict[str, Any] = {"images": len(images), "detections": total_detections}
    if elapsed:
        average = sum(elapsed) / len(elapsed)
        extras["average_inference_ms"] = round(average, 3)
        summary += f"，平均 {average:.1f} ms/张"
        report.log(f"平均推理耗时 {average:.1f} ms/张")
    if args.save:
        extras["output"] = str(save_dir)
        report.artifact("predict_dir", save_dir)
    report.done(summary, **extras)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
