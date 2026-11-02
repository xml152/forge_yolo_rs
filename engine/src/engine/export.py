"""模型导出子命令：``python -m engine.export --weights best.pt --format onnx``。"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from . import report

FORMATS = ["onnx", "torchscript", "openvino", "engine", "coreml", "tflite", "saved_model", "pb"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine.export",
        description="把 .pt 权重导出成部署格式（结果通过 JSON Lines 上报）",
    )
    parser.add_argument("--weights", required=True, help="权重文件（.pt）")
    parser.add_argument("--format", default="onnx", choices=FORMATS, help="目标格式")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False, help="FP16 导出")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False, help="动态输入尺寸")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True, help="简化计算图")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--device", default="", help="导出设备，留空自动")
    return parser


def _resolve(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def run(args: argparse.Namespace) -> int:
    report.quiet()
    from ultralytics import YOLO

    weights = _resolve(args.weights)
    if not weights.is_file():
        report.fail(f"权重文件不存在：{weights}")
        return 2

    report.log(f"开始导出 {weights.name} → {args.format}")
    report.progress("export", 0, 1)

    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "device": args.device or None,
    }
    if args.format == "onnx":
        kwargs["simplify"] = args.simplify
        kwargs["opset"] = args.opset

    try:
        model = YOLO(str(weights))
        output = Path(model.export(**kwargs)).resolve()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report.fail(f"导出失败：{exc}")
        return 1

    report.progress("export", 1, 1)
    report.artifact("export", output, format=args.format)
    report.done(f"导出完成：{output}", path=str(output), format=args.format)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
