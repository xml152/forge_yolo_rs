//! 全局状态：当前页面、共享路径与各页面的编辑状态。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::annotate::Annotator;
use crate::annotate::image::LoadedImage;
use crate::bridge::{self, EngineEvent, Inference, Job, Metrics};
use crate::yolo::convert::{ConvertOptions, TargetFormat};
use crate::yolo::dataset::Dataset;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Page {
    #[default]
    Annotate,
    Train,
    Convert,
    Test,
}

impl Page {
    pub const ALL: [Self; 4] = [Self::Annotate, Self::Train, Self::Convert, Self::Test];

    pub fn title(self) -> &'static str {
        match self {
            Self::Annotate => "数据集标注",
            Self::Train => "模型训练",
            Self::Convert => "格式转换",
            Self::Test => "模型测试",
        }
    }
}

pub struct AppState {
    pub page: Page,
    /// Python 解释器路径（训练/推理时使用）。
    pub python: String,
    pub dataset_root: String,
    pub annotator: Annotator,
    pub train: TrainState,
    pub convert: ConvertState,
    pub test: TestState,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            page: Page::Annotate,
            python: String::new(),
            dataset_root: "datasets".to_string(),
            annotator: Annotator::default(),
            train: TrainState::default(),
            convert: ConvertState::default(),
            test: TestState::default(),
        }
    }

    /// 实际使用的解释器：用户显式填写优先，否则自动查找。
    pub fn resolve_python(&self, engine_root: &Path) -> String {
        let custom = self.python.trim();
        if custom.is_empty() {
            bridge::discover_python(engine_root)
        } else {
            custom.to_string()
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// 数据集格式转换页面状态。
pub struct ConvertState {
    pub dataset_root: String,
    pub out_dir: String,
    pub format: TargetFormat,
    pub val_ratio: f32,
    pub copy_images: bool,
    pub report: String,
}

impl Default for ConvertState {
    fn default() -> Self {
        Self {
            dataset_root: "datasets".to_string(),
            out_dir: "runs/convert".to_string(),
            format: TargetFormat::YoloSplit,
            val_ratio: 0.2,
            copy_images: true,
            report: String::new(),
        }
    }
}

impl ConvertState {
    pub fn run(&mut self) -> Result<String> {
        let dataset = Dataset::open(&self.dataset_root)?;
        if dataset.is_empty() {
            anyhow::bail!("目录中没有找到图片：{}", dataset.root.display());
        }
        let options = ConvertOptions {
            format: self.format,
            val_ratio: self.val_ratio,
            copy_images: self.copy_images,
        };
        let mut messages = Vec::new();
        let report = crate::yolo::convert::convert(
            &dataset,
            Path::new(&self.out_dir),
            &options,
            &mut |message| messages.push(message),
        )?;
        let mut text = report.to_string();
        if !messages.is_empty() {
            text.push('\n');
            for message in messages {
                text.push_str(&message);
                text.push('\n');
            }
        }
        Ok(text)
    }
}

/// 训练页状态：表单参数 + 一个正在运行的训练任务。
pub struct TrainState {
    /// 数据集根目录，用来生成 data.yaml。
    pub dataset_root: String,
    /// 留空则从 `dataset_root` 自动生成 data.yaml。
    pub data_yaml: String,
    /// 留空则用 `engine/weights/yolo26n.pt`。
    pub weights: String,
    pub epochs: u32,
    pub imgsz: u32,
    pub batch: i32,
    pub workers: u32,
    pub device: String,
    pub optimizer: String,
    pub lr0: f32,
    pub patience: u32,
    pub project: String,
    pub name: String,
    pub cache: CacheMode,
    pub amp: bool,
    pub plots: bool,
    pub job: Job,
    /// 每一轮训练结束上报的指标（含最终复评那条）。
    pub history: Vec<Metrics>,
    /// 最近一次上报的指标。
    pub latest: Option<Metrics>,
    pub best: Option<PathBuf>,
    pub last: Option<PathBuf>,
    pub run_dir: Option<PathBuf>,
    pub summary: String,
    pub status: String,
    pub export: ExportState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CacheMode {
    None,
    Ram,
    Disk,
}

impl CacheMode {
    pub const ALL: [Self; 3] = [Self::None, Self::Ram, Self::Disk];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "不缓存",
            Self::Ram => "内存",
            Self::Disk => "磁盘",
        }
    }

    pub fn arg(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ram => "ram",
            Self::Disk => "disk",
        }
    }
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            dataset_root: "datasets".to_string(),
            data_yaml: String::new(),
            weights: String::new(),
            epochs: 100,
            imgsz: 640,
            batch: 16,
            workers: 8,
            device: String::new(),
            optimizer: "auto".to_string(),
            lr0: 0.01,
            patience: 50,
            project: "runs/detect".to_string(),
            name: "train".to_string(),
            cache: CacheMode::None,
            amp: true,
            plots: false,
            job: Job::new("train"),
            history: Vec::new(),
            latest: None,
            best: None,
            last: None,
            run_dir: None,
            summary: String::new(),
            status: String::new(),
            export: ExportState::default(),
        }
    }
}

impl TrainState {
    pub fn start(&mut self, python: &str, engine_root: &Path) -> Result<()> {
        let data = resolve_data_yaml(&self.dataset_root, &self.data_yaml)?;
        let weights = resolve_weights(&self.weights, engine_root);
        self.history.clear();
        self.latest = None;
        self.best = None;
        self.last = None;
        self.run_dir = None;
        self.summary.clear();

        let mut args = vec!["train".to_string(), "--data".to_string(), path_arg(&data)];
        args.extend(pair("--weights", &weights));
        args.extend(pair("--epochs", &self.epochs.to_string()));
        args.extend(pair("--imgsz", &self.imgsz.to_string()));
        args.extend(pair("--batch", &self.batch.to_string()));
        args.extend(pair("--workers", &self.workers.to_string()));
        args.extend(pair("--patience", &self.patience.to_string()));
        args.extend(pair("--optimizer", &self.optimizer));
        args.extend(pair("--lr0", &fmt(self.lr0)));
        args.extend(pair("--project", &self.project));
        args.extend(pair("--name", &self.name));
        args.extend(pair("--cache", self.cache.arg()));
        if !self.device.trim().is_empty() {
            args.extend(pair("--device", self.device.trim()));
        }
        args.push(format!("--{}amp", if self.amp { "" } else { "no-" }));
        args.push(format!("--{}plots", if self.plots { "" } else { "no-" }));

        self.job
            .start(python, engine_root, &args)
            .context("启动训练任务失败")?;
        self.status = "训练已启动".to_string();
        Ok(())
    }

    /// 消费任务事件；返回是否有任务刚刚结束。
    pub fn pump(&mut self) -> bool {
        let update = self.job.pump();
        let finished = update.finished;
        for event in update.events {
            self.apply(event);
        }
        if finished {
            self.status = match self.job.error.as_deref() {
                Some(err) => format!("训练失败：{err}"),
                None => "训练结束".to_string(),
            };
        }
        finished
    }

    fn apply(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Metrics(metrics) => {
                if metrics.is_final {
                    self.latest = Some(metrics.clone());
                } else {
                    self.latest = Some(metrics.clone());
                    self.history.push(metrics);
                }
            }
            EngineEvent::Artifact(artifact) => match artifact.kind.as_str() {
                "best" => {
                    // best.pt 位于 `<project>/<name>/weights/` 下。
                    self.run_dir = artifact
                        .path
                        .parent()
                        .and_then(Path::parent)
                        .map(Path::to_path_buf);
                    self.best = Some(artifact.path);
                }
                "last" => self.last = Some(artifact.path),
                _ => {}
            },
            EngineEvent::Done(done) => {
                // 少数情况下（例如恢复训练）引擎只在 done 里带指标。
                if self.latest.is_none() {
                    self.latest = done.to_metrics();
                }
                self.summary = done.summary;
            }
            _ => {}
        }
    }

    /// 训练产物目录（优先用引擎上报的 save_dir，否则用 project/name）。
    pub fn output_dir(&self) -> PathBuf {
        self.run_dir
            .clone()
            .unwrap_or_else(|| Path::new(&self.project).join(&self.name))
    }
}

/// 模型导出（ONNX 等）子状态。
pub struct ExportState {
    /// 留空则跟随训练得到的最佳权重。
    pub weights: String,
    pub format: String,
    pub imgsz: u32,
    pub half: bool,
    pub dynamic: bool,
    pub simplify: bool,
    pub opset: u32,
    pub job: Job,
    pub output: Option<PathBuf>,
    pub status: String,
}

/// 与 `engine/src/engine/export.py` 的 `FORMATS` 保持一致。
pub const EXPORT_FORMATS: &[&str] = &[
    "onnx",
    "torchscript",
    "openvino",
    "engine",
    "coreml",
    "tflite",
    "saved_model",
    "pb",
];

impl Default for ExportState {
    fn default() -> Self {
        Self {
            weights: String::new(),
            format: "onnx".to_string(),
            imgsz: 640,
            half: false,
            dynamic: false,
            simplify: true,
            opset: 17,
            job: Job::new("export"),
            output: None,
            status: String::new(),
        }
    }
}

impl ExportState {
    pub fn start(
        &mut self,
        python: &str,
        engine_root: &Path,
        fallback: Option<&Path>,
    ) -> Result<()> {
        let weights = if self.weights.trim().is_empty() {
            match fallback {
                Some(path) => path.display().to_string(),
                None => anyhow::bail!("请先指定权重文件，或先训练出一个 best.pt"),
            }
        } else {
            self.weights.trim().to_string()
        };
        self.output = None;

        let mut args = vec!["export".to_string()];
        args.extend(pair("--weights", &weights));
        args.extend(pair("--format", &self.format));
        args.extend(pair("--imgsz", &self.imgsz.to_string()));
        args.extend(pair("--opset", &self.opset.to_string()));
        args.push(format!("--{}half", if self.half { "" } else { "no-" }));
        args.push(format!(
            "--{}dynamic",
            if self.dynamic { "" } else { "no-" }
        ));
        args.push(format!(
            "--{}simplify",
            if self.simplify { "" } else { "no-" }
        ));

        self.job
            .start(python, engine_root, &args)
            .context("启动导出任务失败")?;
        self.status = "导出已启动".to_string();
        Ok(())
    }

    pub fn pump(&mut self) -> bool {
        let update = self.job.pump();
        let finished = update.finished;
        for event in update.events {
            match event {
                EngineEvent::Artifact(artifact) => self.output = Some(artifact.path),
                EngineEvent::Done(done) => {
                    if let Some(path) = done.field("path") {
                        self.output = Some(PathBuf::from(path));
                    }
                    self.status = done.summary;
                }
                _ => {}
            }
        }
        if finished && self.job.error.is_some() {
            self.status = format!("导出失败：{}", self.job.error.as_deref().unwrap_or(""));
        }
        finished
    }
}

/// 测试页状态：推理预览 + 指标评估。
pub struct TestState {
    pub weights: String,
    pub source: String,
    pub imgsz: u32,
    pub conf: f32,
    pub iou: f32,
    pub device: String,
    pub max_det: u32,
    pub save: bool,
    pub txt: bool,
    pub project: String,
    pub name: String,
    pub job: Job,
    pub results: Vec<Inference>,
    pub selected: Option<usize>,
    pub preview: PreviewImage,
    pub summary: String,
    pub status: String,
    /// 评估子任务
    pub dataset_root: String,
    pub data_yaml: String,
    pub split: String,
    pub val_job: Job,
    pub val_metrics: Option<Metrics>,
    pub val_summary: String,
    pub val_status: String,
}

impl Default for TestState {
    fn default() -> Self {
        Self {
            weights: String::new(),
            source: String::new(),
            imgsz: 640,
            conf: 0.25,
            iou: 0.45,
            device: String::new(),
            max_det: 300,
            save: true,
            txt: false,
            project: "runs/predict".to_string(),
            name: "test".to_string(),
            job: Job::new("infer"),
            results: Vec::new(),
            selected: None,
            preview: PreviewImage::default(),
            summary: String::new(),
            status: String::new(),
            dataset_root: "datasets".to_string(),
            data_yaml: String::new(),
            split: "val".to_string(),
            val_job: Job::new("val"),
            val_metrics: None,
            val_summary: String::new(),
            val_status: String::new(),
        }
    }
}

/// 模型测试页的 `--split` 取值，与 `val.py` 的 choices 一致。
pub const VAL_SPLITS: &[&str] = &["train", "val", "test"];

impl TestState {
    pub fn start(&mut self, python: &str, engine_root: &Path) -> Result<()> {
        let weights = require_file(&self.weights, "权重文件")?;
        let source = require_file(&self.source, "推理图片或目录")?;
        self.results.clear();
        self.selected = None;
        self.preview.clear();
        self.summary.clear();

        let mut args = vec!["infer".to_string()];
        args.extend(pair("--weights", &weights));
        args.extend(pair("--source", &source));
        args.extend(pair("--imgsz", &self.imgsz.to_string()));
        args.extend(pair("--conf", &fmt(self.conf)));
        args.extend(pair("--iou", &fmt(self.iou)));
        args.extend(pair("--max-det", &self.max_det.to_string()));
        args.extend(pair("--project", &self.project));
        args.extend(pair("--name", &self.name));
        if !self.device.trim().is_empty() {
            args.extend(pair("--device", self.device.trim()));
        }
        args.push(format!("--{}save", if self.save { "" } else { "no-" }));
        args.push(format!("--{}txt", if self.txt { "" } else { "no-" }));

        self.job
            .start(python, engine_root, &args)
            .context("启动推理任务失败")?;
        self.status = "推理已启动".to_string();
        Ok(())
    }

    pub fn pump(&mut self) -> bool {
        let update = self.job.pump();
        let finished = update.finished;
        for event in update.events {
            match event {
                EngineEvent::Result(inference) => self.results.push(inference),
                EngineEvent::Done(done) => self.summary = done.summary,
                _ => {}
            }
        }
        if finished {
            self.status = match self.job.error.as_deref() {
                Some(err) => format!("推理失败：{err}"),
                None => "推理结束".to_string(),
            };
        }
        finished
    }

    pub fn start_val(&mut self, python: &str, engine_root: &Path) -> Result<()> {
        let weights = require_file(&self.weights, "权重文件")?;
        let data = resolve_data_yaml(&self.dataset_root, &self.data_yaml)?;
        self.val_metrics = None;
        self.val_summary.clear();

        let mut args = vec!["val".to_string()];
        args.extend(pair("--weights", &weights));
        args.extend(pair("--data", &path_arg(&data)));
        args.extend(pair("--split", &self.split));
        args.extend(pair("--imgsz", &self.imgsz.to_string()));
        if !self.device.trim().is_empty() {
            args.extend(pair("--device", self.device.trim()));
        }
        args.push("--no-plots".to_string());

        self.val_job
            .start(python, engine_root, &args)
            .context("启动评估任务失败")?;
        self.val_status = "评估已启动".to_string();
        Ok(())
    }

    pub fn pump_val(&mut self) -> bool {
        let update = self.val_job.pump();
        let finished = update.finished;
        for event in update.events {
            match event {
                EngineEvent::Metrics(metrics) => self.val_metrics = Some(metrics),
                EngineEvent::Done(done) => self.val_summary = done.summary,
                _ => {}
            }
        }
        if finished {
            self.val_status = match self.val_job.error.as_deref() {
                Some(err) => format!("评估失败：{err}"),
                None => "评估结束".to_string(),
            };
        }
        finished
    }

    pub fn selected_result(&self) -> Option<&Inference> {
        self.selected.and_then(|index| self.results.get(index))
    }
}

/// 一张按需加载的预览图（带框标注图等）。
#[derive(Default)]
pub struct PreviewImage {
    path: Option<PathBuf>,
    image: Option<LoadedImage>,
    pub error: Option<String>,
}

impl PreviewImage {
    pub fn show(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        if self.path.as_deref() != Some(path.as_path()) {
            self.path = Some(path);
            self.image = None;
            self.error = None;
        }
    }

    pub fn clear(&mut self) {
        self.path = None;
        self.image = None;
        self.error = None;
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// 每帧调用：按需解码当前路径的图片。
    pub fn ensure(&mut self, ctx: &egui::Context) -> Option<&LoadedImage> {
        if self.image.is_none() && self.error.is_none() {
            let path = self.path.clone()?;
            match LoadedImage::load(ctx, &path) {
                Ok(image) => self.image = Some(image),
                Err(err) => self.error = Some(format!("{err:#}")),
            }
        }
        self.image.as_ref()
    }
}

/// 留空即自动生成的 data.yaml 路径。
fn resolve_data_yaml(dataset_root: &str, explicit: &str) -> Result<PathBuf> {
    let explicit = explicit.trim();
    if !explicit.is_empty() {
        let path = PathBuf::from(explicit);
        if !path.is_file() {
            anyhow::bail!("data.yaml 不存在：{}", path.display());
        }
        return Ok(path);
    }
    let root = Path::new(dataset_root.trim());
    if !root.is_dir() {
        anyhow::bail!("数据集目录不存在：{}", root.display());
    }
    let existing = root.join("data.yaml");
    if existing.is_file() {
        return Ok(existing);
    }
    let dataset = Dataset::open(root)?;
    if dataset.is_empty() {
        anyhow::bail!("目录中没有找到图片：{}", dataset.root.display());
    }
    // 引擎子进程的 cwd 是 engine/，因此写绝对路径最稳妥。
    let cache = Path::new("runs").join("forge-data.yaml");
    if let Some(parent) = cache.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cache, dataset.data_yaml(false))
        .with_context(|| format!("写入 {} 失败", cache.display()))?;
    std::fs::canonicalize(&cache).context("无法解析自动生成的 data.yaml 路径")
}

fn resolve_weights(weights: &str, engine_root: &Path) -> String {
    let weights = weights.trim();
    if !weights.is_empty() {
        return weights.to_string();
    }
    let candidate = engine_root.join("weights").join("yolo26n.pt");
    if candidate.is_file() {
        path_arg(&candidate)
    } else {
        "yolo26n.pt".to_string()
    }
}

fn require_file(value: &str, what: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        anyhow::bail!("请先选择{what}");
    }
    let path = Path::new(value);
    if !path.exists() {
        anyhow::bail!("{what}不存在：{}", path.display());
    }
    Ok(path_arg(path))
}

fn path_arg(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn pair(flag: &str, value: &str) -> [String; 2] {
    [flag.to_string(), value.to_string()]
}

fn fmt(value: f32) -> String {
    let mut text = format!("{value}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
}

/// 需要跨进程保存的界面配置。
///
/// `AppState` 里含有子进程句柄、图片缓存与 GPU 纹理等无法序列化的东西，
/// 所以持久化只走这份"表单字段"快照；字段缺省时回落到各自的默认值。
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    pub page: Page,
    pub python: String,
    pub annotate: AnnotateSetting,
    pub train: TrainSetting,
    pub convert: ConvertSetting,
    pub test: TestSetting,
}

impl Settings {
    /// 从当前界面状态抽出一份快照。
    pub fn capture(state: &AppState) -> Self {
        Self {
            page: state.page,
            python: state.python.clone(),
            annotate: AnnotateSetting {
                dataset_root: state.dataset_root.clone(),
            },
            train: TrainSetting {
                dataset_root: state.train.dataset_root.clone(),
                data_yaml: state.train.data_yaml.clone(),
                weights: state.train.weights.clone(),
                epochs: state.train.epochs,
                imgsz: state.train.imgsz,
                batch: state.train.batch,
                workers: state.train.workers,
                device: state.train.device.clone(),
                optimizer: state.train.optimizer.clone(),
                lr0: state.train.lr0,
                patience: state.train.patience,
                project: state.train.project.clone(),
                name: state.train.name.clone(),
                cache: state.train.cache,
                amp: state.train.amp,
                plots: state.train.plots,
                export: ExportSetting::from_state(&state.train.export),
            },
            convert: ConvertSetting {
                dataset_root: state.convert.dataset_root.clone(),
                out_dir: state.convert.out_dir.clone(),
                format: state.convert.format,
                val_ratio: state.convert.val_ratio,
                copy_images: state.convert.copy_images,
            },
            test: TestSetting {
                weights: state.test.weights.clone(),
                source: state.test.source.clone(),
                imgsz: state.test.imgsz,
                conf: state.test.conf,
                iou: state.test.iou,
                device: state.test.device.clone(),
                max_det: state.test.max_det,
                save: state.test.save,
                txt: state.test.txt,
                project: state.test.project.clone(),
                name: state.test.name.clone(),
                dataset_root: state.test.dataset_root.clone(),
                data_yaml: state.test.data_yaml.clone(),
                split: state.test.split.clone(),
            },
        }
    }

    /// 把快照写回界面状态；运行中的任务与结果不受影响。
    pub fn apply(self, state: &mut AppState) {
        state.page = self.page;
        state.python = self.python;
        state.dataset_root = self.annotate.dataset_root;

        let train = &mut state.train;
        train.dataset_root = self.train.dataset_root;
        train.data_yaml = self.train.data_yaml;
        train.weights = self.train.weights;
        train.epochs = self.train.epochs;
        train.imgsz = self.train.imgsz;
        train.batch = self.train.batch;
        train.workers = self.train.workers;
        train.device = self.train.device;
        train.optimizer = self.train.optimizer;
        train.lr0 = self.train.lr0;
        train.patience = self.train.patience;
        train.project = self.train.project;
        train.name = self.train.name;
        train.cache = self.train.cache;
        train.amp = self.train.amp;
        train.plots = self.train.plots;
        train.export.weights = self.train.export.weights;
        train.export.format = self.train.export.format;
        train.export.imgsz = self.train.export.imgsz;
        train.export.half = self.train.export.half;
        train.export.dynamic = self.train.export.dynamic;
        train.export.simplify = self.train.export.simplify;
        train.export.opset = self.train.export.opset;

        let convert = &mut state.convert;
        convert.dataset_root = self.convert.dataset_root;
        convert.out_dir = self.convert.out_dir;
        convert.format = self.convert.format;
        convert.val_ratio = self.convert.val_ratio;
        convert.copy_images = self.convert.copy_images;

        let test = &mut state.test;
        test.weights = self.test.weights;
        test.source = self.test.source;
        test.imgsz = self.test.imgsz;
        test.conf = self.test.conf;
        test.iou = self.test.iou;
        test.device = self.test.device;
        test.max_det = self.test.max_det;
        test.save = self.test.save;
        test.txt = self.test.txt;
        test.project = self.test.project;
        test.name = self.test.name;
        test.dataset_root = self.test.dataset_root;
        test.data_yaml = self.test.data_yaml;
        test.split = self.test.split;
    }
}

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AnnotateSetting {
    pub dataset_root: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TrainSetting {
    pub dataset_root: String,
    pub data_yaml: String,
    pub weights: String,
    pub epochs: u32,
    pub imgsz: u32,
    pub batch: i32,
    pub workers: u32,
    pub device: String,
    pub optimizer: String,
    pub lr0: f32,
    pub patience: u32,
    pub project: String,
    pub name: String,
    pub cache: CacheMode,
    pub amp: bool,
    pub plots: bool,
    pub export: ExportSetting,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ExportSetting {
    pub weights: String,
    pub format: String,
    pub imgsz: u32,
    pub half: bool,
    pub dynamic: bool,
    pub simplify: bool,
    pub opset: u32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ConvertSetting {
    pub dataset_root: String,
    pub out_dir: String,
    pub format: TargetFormat,
    pub val_ratio: f32,
    pub copy_images: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TestSetting {
    pub weights: String,
    pub source: String,
    pub imgsz: u32,
    pub conf: f32,
    pub iou: f32,
    pub device: String,
    pub max_det: u32,
    pub save: bool,
    pub txt: bool,
    pub project: String,
    pub name: String,
    pub dataset_root: String,
    pub data_yaml: String,
    pub split: String,
}

impl Default for TrainSetting {
    fn default() -> Self {
        let d = TrainState::default();
        Self {
            dataset_root: d.dataset_root,
            data_yaml: d.data_yaml,
            weights: d.weights,
            epochs: d.epochs,
            imgsz: d.imgsz,
            batch: d.batch,
            workers: d.workers,
            device: d.device,
            optimizer: d.optimizer,
            lr0: d.lr0,
            patience: d.patience,
            project: d.project,
            name: d.name,
            cache: d.cache,
            amp: d.amp,
            plots: d.plots,
            export: ExportSetting::from_state(&d.export),
        }
    }
}

impl ExportSetting {
    fn from_state(export: &ExportState) -> Self {
        Self {
            weights: export.weights.clone(),
            format: export.format.clone(),
            imgsz: export.imgsz,
            half: export.half,
            dynamic: export.dynamic,
            simplify: export.simplify,
            opset: export.opset,
        }
    }
}

impl Default for ExportSetting {
    fn default() -> Self {
        Self::from_state(&ExportState::default())
    }
}

impl Default for ConvertSetting {
    fn default() -> Self {
        let d = ConvertState::default();
        Self {
            dataset_root: d.dataset_root,
            out_dir: d.out_dir,
            format: d.format,
            val_ratio: d.val_ratio,
            copy_images: d.copy_images,
        }
    }
}

impl Default for TestSetting {
    fn default() -> Self {
        let d = TestState::default();
        Self {
            weights: d.weights,
            source: d.source,
            imgsz: d.imgsz,
            conf: d.conf,
            iou: d.iou,
            device: d.device,
            max_det: d.max_det,
            save: d.save,
            txt: d.txt,
            project: d.project,
            name: d.name,
            dataset_root: d.dataset_root,
            data_yaml: d.data_yaml,
            split: d.split,
        }
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn capture_and_apply_roundtrip() {
        let mut state = AppState::new();
        state.page = Page::Test;
        state.python = "/opt/py/bin/python".to_string();
        state.dataset_root = "datasets/gesture".to_string();
        state.train.epochs = 7;
        state.train.imgsz = 320;
        state.train.batch = -1;
        state.train.device = "0".to_string();
        state.train.optimizer = "AdamW".to_string();
        state.train.lr0 = 0.001;
        state.train.patience = 12;
        state.train.cache = CacheMode::Ram;
        state.train.amp = false;
        state.train.plots = true;
        state.train.export.format = "openvino".to_string();
        state.train.export.opset = 13;
        state.convert.format = TargetFormat::Coco;
        state.convert.val_ratio = 0.35;
        state.convert.copy_images = false;
        state.test.source = "datasets/gesture/images/val".to_string();
        state.test.conf = 0.6;
        state.test.split = "test".to_string();

        let json = serde_json::to_string(&Settings::capture(&state)).unwrap();
        let restored: Settings = serde_json::from_str(&json).unwrap();

        let mut target = AppState::new();
        restored.apply(&mut target);

        assert_eq!(target.page, Page::Test);
        assert_eq!(target.python, "/opt/py/bin/python");
        assert_eq!(target.dataset_root, "datasets/gesture");
        assert_eq!(target.train.epochs, 7);
        assert_eq!(target.train.imgsz, 320);
        assert_eq!(target.train.batch, -1);
        assert_eq!(target.train.device, "0");
        assert_eq!(target.train.optimizer, "AdamW");
        assert!((target.train.lr0 - 0.001).abs() < 1e-9);
        assert_eq!(target.train.patience, 12);
        assert_eq!(target.train.cache, CacheMode::Ram);
        assert!(!target.train.amp);
        assert!(target.train.plots);
        assert_eq!(target.train.export.format, "openvino");
        assert_eq!(target.train.export.opset, 13);
        assert_eq!(target.convert.format, TargetFormat::Coco);
        assert!((target.convert.val_ratio - 0.35).abs() < 1e-6);
        assert!(!target.convert.copy_images);
        assert_eq!(target.test.source, "datasets/gesture/images/val");
        assert!((target.test.conf - 0.6).abs() < 1e-6);
        assert_eq!(target.test.split, "test");
    }

    #[test]
    fn empty_json_falls_back_to_defaults() {
        // 旧版本存档缺字段时必须能读起来，而不是整体失败。
        let settings: Settings = serde_json::from_str("{}").unwrap();
        let mut state = AppState::new();
        settings.apply(&mut state);

        let default = AppState::new();
        assert_eq!(state.page, default.page);
        assert_eq!(state.train.epochs, default.train.epochs);
        assert_eq!(state.train.cache, default.train.cache);
        assert_eq!(state.convert.format, default.convert.format);
        assert_eq!(state.test.imgsz, default.test.imgsz);
    }

    #[test]
    fn partial_json_keeps_provided_values() {
        let settings: Settings =
            serde_json::from_str(r#"{"python":"/usr/bin/python3","train":{"epochs":3}}"#).unwrap();
        let mut state = AppState::new();
        settings.apply(&mut state);

        assert_eq!(state.python, "/usr/bin/python3");
        assert_eq!(state.train.epochs, 3);
        // 未提供的字段用默认值，不受影响。
        assert_eq!(state.train.imgsz, TrainState::default().imgsz);
    }
}
