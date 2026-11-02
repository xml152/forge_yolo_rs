# Forge YOLO

一个用 Rust 写的 YOLO 桌面工具，覆盖 **标注 → 训练 → 评估 → 推理 → 格式转换** 全流程。

- **上层 UI 与数据处理**：Rust + egui/eframe（单进程、原生窗口）。
- **底层训练/推理**：Python + ultralytics，以子进程方式按需拉起。
- **两侧通信**：`stdout` 上的 JSON Lines 协议，前端逐帧轮询解析。

Rust 负责所有交互与文件读写，Python 只管“跑模型”，因此训练侧的依赖（torch/CUDA）不会污染 Rust 构建。

---

## 目录结构

```
forge_yolo_rs/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs              # 入口：初始化 eframe 窗口
│   ├── app.rs               # ForgeApp，实现 eframe::App（含中文字体装载）
│   ├── state.rs             # 全局状态与各页面状态机（训练/评估/导出的编排都在这里）
│   ├── ui/
│   │   ├── mod.rs           # 顶栏 + 页面分派 + 共享的 class_color()
│   │   ├── annotate.rs      # 数据集标注页
│   │   ├── train.rs         # 模型训练页（含导出区）
│   │   ├── convert.rs       # 格式转换页
│   │   └── test.rs          # 模型测试页（推理 + 评估）
│   ├── annotate/            # 标注画布核心
│   │   ├── mod.rs           # Annotator：打开数据集、翻页、增删改框、撤销、保存
│   │   ├── image.rs         # 图片解码与 egui 纹理缓存
│   │   └── bbox.rs          # 框的命中测试与拖拽（移动 / 8 角缩放）
│   ├── yolo/                # YOLO 数据格式
│   │   ├── dataset.rs       # 数据集扫描、classes.txt、data.yaml 生成
│   │   ├── label.rs         # YOLO txt 的解析与序列化
│   │   └── convert.rs       # 导出为 YOLO split / COCO / VOC
│   └── bridge/              # Rust ↔ Python 桥接
│       ├── mod.rs           # 引擎目录发现、解释器发现、Job 任务对象
│       ├── process.rs       # 子进程生命周期、管道读取、取消
│       └── protocol.rs      # 协议类型与手写 JSON 解析
└── engine/                  # Python 引擎（uv 管理）
    ├── pyproject.toml
    ├── weights/             # 权重（yolo26n.pt、训练产出的 best.pt 等）
    ├── datasets/            # 数据集
    ├── runs/                # ultralytics 输出（训练/推理/评估）
    ├── src/engine/
    │   ├── __init__.py      # 子命令分发
    │   ├── __main__.py      # 支持 python -m engine
    │   ├── report.py        # JSON Lines 上报（stdout 只走协议）
    │   ├── metric_utils.py  # 逐类指标抽取（train / val 共用）
    │   ├── train.py
    │   ├── infer.py
    │   ├── val.py
    │   └── export.py
    └── tests/               # 引擎单元测试（标准库 unittest）
```

---

## 环境搭建

### 1. Rust 侧

```bash
cargo build --release
```

需要系统具备 OpenGL/GL 或 Vulkan 驱动（eframe 只启用了 `glow`）。

### 2. Python 引擎侧

引擎用 [uv](https://docs.astral.sh/uv/) 管理，且以 **editable** 方式安装，这样 `python -m engine` 才能找到包：

```bash
cd engine
uv sync                 # 创建 .venv 并安装 ultralytics
uv pip install -e .     # 若 uv sync 未自动安装自身，手动做一次
```

不用 uv 的话：

```bash
cd engine
python3 -m venv .venv
.venv/bin/pip install -e .
```

装好后自检：

```bash
cd engine
.venv/bin/python -m engine          # 列出可用子命令
.venv/bin/python -m engine train -h
```

> **CUDA**：GPU 训练/推理依赖 ultralytics 拉来的 torch 版本。若需要特定 CUDA 版本，先在 `engine/` 下按 PyTorch 官网命令装 torch，再装 ultralytics。

---

## 运行

```bash
cargo run --release
```

启动后顶栏右侧有「Python」输入框，留空即自动发现：

1. `engine/.venv/bin/python`
2. `PATH` 里的 `python3`
3. `PATH` 里的 `python`

**引擎目录**同样会自动发现，顺序为：

1. 环境变量 `FORGE_ENGINE_ROOT`（指向包含 `src/engine/__init__.py` 的目录）
2. 当前工作目录下的 `engine/`
3. 当前工作目录本身（若它就是个引擎目录）
4. 可执行文件向上 4 层祖先目录下的 `engine/`

子进程的 `cwd` 会被设为引擎目录，因此 `weights/`、`runs/` 这类相对路径始终相对 `engine/`。UI 里选择文件时会传绝对路径，两边都不容易踩坑。

### 配置会记住

所有表单字段（解释器路径、数据集目录、训练超参、导出格式、推理/评估参数、当前页面）都会自动存到系统配置目录：

- Linux：`~/.local/share/forgeyolo/app.ron`
- macOS：`~/Library/Application Support/forgeyolo/app.ron`
- Windows：`%APPDATA%\forgeyolo\app.ron`

退出时以及运行中每 10 秒各写一次。存档带 `#[serde(default)]`，新增字段不会让旧存档失效；删掉该文件即恢复出厂默认值。运行中的任务、图片缓存、推理结果不参与持久化。

---

## 四个页面

### 数据集标注

- 「打开」填数据集根目录（含 `images/`、`labels/`、`classes.txt`），「浏览…」用文件对话框选择。
- 画布上拖拽空白处新建框；点击选中已有框；拖拽框体移动；拖拽 8 个角/边做缩放。
- 快捷键：`A`/`D` 上一张/下一张，`N` 新建框，`Del` 删除选中框，`Ctrl+S` 保存，`Ctrl+Z` 撤销。
- 类别面板显示 `classes.txt` 里的类名与配色（颜色由类别 id 经黄金分割色相生成，稳定不变）。
- 底部统计图片数 / 已标注数 / 总框数，并显示标注文件的实际目录。

### 模型训练

- 必填：权重、数据集 `data.yaml`（可从 UI 选择数据集后自动生成，写入 `runs/forge-data.yaml`）。
- 可调：epochs、imgsz、batch、workers、device、patience、optimizer、lr0、cache、AMP、plots、resume、seed。
- 运行中显示分阶段进度条（训练轮次 / batch）与实时日志（保留最近 5000 行，自动滚到底）。
- 指标面板：每轮的 `loss`、`precision`、`recall`、`mAP50`、`mAP50-95` 与学习率；逐类别 `mAP50-95` 按 3 列平铺，类名着色。
- 一次训练结束时会额外收到一条 `final=true` 的指标（ultralytics 的复评结果），用它兜底显示最终数值。
- 训练页尾部的**导出区**可把 `best.pt` 导出为 ONNX / TorchScript / OpenVINO / Engine / CoreML / TFLite / SavedModel / PB，可设 imgsz、half、dynamic、simplify、opset。

### 格式转换

把 YOLO 数据集另存为三种布局之一：

- `YoloSplit`：按 `train/val` 重新切分
- `Coco`：单个 `annotations.json`
- `Voc`：每张图一个 XML

可设验证集比例与随机种子（切分是确定性的）。完成后报告图片数、框数与跳过项。

### 模型测试

一个页面分两块：

- **推理**：选权重与图片目录，设 imgsz/conf/iou/device/max-det，逐张上报结果。左侧结果列表点击切换图片，右侧显示带框预览；鼠标悬停列表项可看每个框的归一化 `xywhn`。底部汇总图片数、目标数、平均推理耗时。
- **评估**：选权重、`data.yaml` 与 `split`（train/val/test），只跑指标不存图。展示整体 `precision`/`recall`/`mAP50`/`mAP50-95`、各阶段耗时，以及逐类别 `mAP50-95`。

---

## Rust ↔ Python 协议

Python 侧 `engine/src/engine/report.py` 在导入时就把 `sys.stdout` 换成 `sys.stderr`，只在内部保留原始 stdout 用于发协议。因此 **stdout 上只有 JSON 事件**，ultralytics 的日志、TQDM、第三方 `print` 全部走 stderr，前端把它们当普通日志收。

每行一个 JSON 对象：

| `type` | 关键字段 | 含义 |
| --- | --- | --- |
| `log` | `level`, `message` | 普通日志（`info` / `warn`） |
| `progress` | `stage`, `current`, `total`, `epoch`?, `epochs`?, `percent`? | 进度；`stage` ∈ `epoch`/`batch`/`predict`/`val`/`export` |
| `metrics` | `epoch`, `epochs`, `values`, `final`? | 每轮指标；`final=true` 表示复评结果 |
| `artifact` | `kind`, `path` | 产物路径；`kind` ∈ `best`/`last`/`predict_dir`/`val_dir`/`export` |
| `result` | `payload` | 单张推理结果（`image`、`width`、`height`、`detections`、`speed`、`annotated`?） |
| `done` | `summary` + 各命令自有字段 | 结束汇总 |
| `error` | `message` | 失败原因 |

指标键名两套并存：训练回调上报的是 ultralytics 原始键（`metrics/mAP50-95(B)`），评估命令上报的是裸键（`mAP50-95`）。前端的 `Metrics::lookup` 会先做全体精确匹配、再做全体前缀匹配，因此两种写法都能取到。

### 命令行等价用法

UI 背后就是这些命令，脚本化调试时可直接跑：

```bash
cd engine

python -m engine train  --data ../runs/forge-data.yaml --weights weights/yolo26n.pt --epochs 100
python -m engine infer  --weights runs/detect/train/weights/best.pt --source datasets/hands_gesture/images/val
python -m engine val    --weights runs/detect/train/weights/best.pt --data datasets/hands_gesture/data.yaml --split val
python -m engine export --weights runs/detect/train/weights/best.pt --format onnx --imgsz 640
```

各子命令的完整参数用 `-h` 查看。

---

## 数据集格式

```
datasets/hands_gesture/
├── classes.txt              # 每行一个类名，行号即类别 id
├── data.yaml                # 可手写；缺字段时 UI 会自动生成
├── images/
│   ├── train/*.png
│   └── val/*.png
└── labels/
    ├── train/*.txt          # 与图片同名，每行 `class_id cx cy w h`（归一化）
    └── val/*.txt
```

`data.yaml` 由 `Dataset::data_yaml()` 生成时会处理划分式布局、把路径相对化，并补齐 `nc:` 字段——ultralytics 的 `check_det_dataset` 要求 `names`/`nc` 至少存在一个且彼此一致。

---

## 开发

```bash
cargo check --all-targets   # 类型检查（应零警告）
cargo test                  # 单元测试（28 个）
cargo test -- --ignored     # 端到端测试（真实引擎跑 val / infer / export / train，约 35 秒）
cargo fmt                   # 格式化

# Python 引擎的单元测试
cd engine && .venv/bin/python -m unittest discover -s tests -v
```

单元测试覆盖：协议解析（含 `per_class` 嵌套字典）、标注框拖拽几何、YOLO 标签读写、`data.yaml` 解析、格式转换切分确定性、引擎目录发现、四页 UI 无头渲染、界面配置的序列化往返。

`--ignored` 的端到端测试会真的启动 Python 子进程，校验完整的 spawn → JSON Lines 解析 → `Job` 状态机链路，并断言真实指标字段存在（如 `mAP50-95`、逐类 `per_class`）。需要 `engine/.venv/` 已就绪，训练用例还会写出 `engine/runs/e2e-train/`（测试结束自行删除）。

临时产物（`engine/.venv/`、`engine/datasets/`、`engine/runs/`、`/runs/`、`target/`）已在 `.gitignore` 中排除。
