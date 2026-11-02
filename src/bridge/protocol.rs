//! Rust 与 Python 引擎之间的 JSON Lines 协议。
//!
//! Python 侧每行输出一个 JSON 对象，形如 `{"type": "log", ...}`。这里用手写解析
//! 而不是 serde 派生，因为协议里的 `metrics`/`done` 会携带任意附加字段，
//! 手写解析对字段增减更宽容：认不出的字段直接忽略，认不出的 `type` 返回 `None`。

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{Map, Value};

/// Python 侧 `report.log` 的日志级别。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn parse(text: &str) -> Self {
        match text {
            "warn" | "warning" => Self::Warn,
            "error" | "critical" => Self::Error,
            _ => Self::Info,
        }
    }
}

/// 一条协议事件。
#[derive(Clone, Debug)]
pub enum EngineEvent {
    Log { level: LogLevel, message: String },
    Progress(Progress),
    Metrics(Metrics),
    Artifact(Artifact),
    Result(Inference),
    Done(Done),
    Error(String),
}

/// `{"type":"progress","stage":..,"current":..,"total":..}`
#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub stage: String,
    pub current: f64,
    pub total: f64,
    /// `stage == "epoch"` 时携带：本轮序号（从 1 开始）与总轮数。
    pub epoch: Option<u32>,
    pub epochs: Option<u32>,
    /// `stage == "batch"` 时携带的批进度（0..=1）。
    pub percent: Option<f32>,
}

impl Progress {
    pub fn ratio(&self) -> f32 {
        if self.total <= 0.0 {
            return 0.0;
        }
        (self.current / self.total).clamp(0.0, 1.0) as f32
    }

    pub fn text(&self) -> String {
        match (self.epoch, self.epochs) {
            (Some(epoch), Some(epochs)) => format!("第 {epoch}/{epochs} 轮"),
            _ => format!("{}/{}", fmt_num(self.current), fmt_num(self.total)),
        }
    }
}

/// `{"type":"metrics","epoch":..,"epochs":..,"values":{...}}`
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub epoch: u32,
    pub epochs: u32,
    /// 标量指标（loss、precision、mAP、学习率……）。
    pub values: BTreeMap<String, f64>,
    /// `values.per_class`：验证集上每个类别的 mAP50-95。
    pub per_class: Vec<ClassMetric>,
    /// 训练结束前 ultralytics 会用最佳权重再评估一次，这条标记为 `true`。
    pub is_final: bool,
}

impl Metrics {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.values.get(key).copied()
    }

    /// 按候选键取指标。先在所有候选里找完全相同的键，再退回前缀匹配
    /// （ultralytics 的键名常带后缀，如 `metrics/mAP50(B)`）。
    pub fn lookup(&self, candidates: &[&str]) -> Option<f64> {
        for key in candidates {
            if let Some(value) = self.get(key) {
                return Some(value);
            }
        }
        for key in candidates {
            if let Some((_, value)) = self.values.iter().find(|(name, _)| name.starts_with(key)) {
                return Some(*value);
            }
        }
        None
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClassMetric {
    pub class_id: u32,
    pub name: String,
    pub map: f64,
}

/// `{"type":"artifact","kind":"best","path":"..."}`
#[derive(Clone, Debug)]
pub struct Artifact {
    pub kind: String,
    pub path: PathBuf,
}

/// 单个检测框。
#[derive(Clone, Debug, Default)]
pub struct Detection {
    pub class_id: u32,
    pub name: String,
    pub confidence: f32,
    /// `[x1, y1, x2, y2]`，像素坐标。
    pub xyxy: [f32; 4],
    /// `[cx, cy, w, h]`，归一化坐标。
    pub xywhn: [f32; 4],
}

/// `{"type":"result","payload":{...}}`：一张图片的推理结果。
#[derive(Clone, Debug, Default)]
pub struct Inference {
    pub image: PathBuf,
    pub width: u32,
    pub height: u32,
    pub detections: Vec<Detection>,
    pub speed: BTreeMap<String, f64>,
    pub annotated: Option<PathBuf>,
}

impl Inference {
    pub fn file_name(&self) -> String {
        self.image
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    /// 单张图推理耗时（毫秒），取 ultralytics 的 `speed.inference`。
    pub fn inference_ms(&self) -> Option<f64> {
        self.speed.get("inference").copied()
    }
}

/// `{"type":"done","summary":"..."}`：一次任务的收尾事件。
#[derive(Clone, Debug, Default)]
pub struct Done {
    pub summary: String,
    /// 除 `type`/`summary`/`metrics` 之外的附加字段，已格式化为可直接显示的文本。
    pub fields: Vec<(String, String)>,
    pub metrics: BTreeMap<String, f64>,
    pub per_class: Vec<ClassMetric>,
}

impl Done {
    pub fn field(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }

    /// 收尾事件里带的指标转成一条 [`Metrics`]；没有指标时返回 `None`。
    ///
    /// 大部分子命令会先发 `metrics` 再发 `done`，但也有只发 `done` 的情况
    /// （例如恢复训练），调用方可拿它做兜底。
    pub fn to_metrics(&self) -> Option<Metrics> {
        if self.metrics.is_empty() && self.per_class.is_empty() {
            return None;
        }
        Some(Metrics {
            values: self.metrics.clone(),
            per_class: self.per_class.clone(),
            is_final: true,
            ..Default::default()
        })
    }
}

/// 解析一行协议输出；不是协议事件时返回 `None`。
pub fn parse_line(line: &str) -> Option<EngineEvent> {
    let line = line.trim();
    if !line.starts_with('{') {
        return None;
    }
    let value: Value = serde_json::from_str(line).ok()?;
    let obj = value.as_object()?;
    let kind = obj.get("type")?.as_str()?;

    Some(match kind {
        "log" => EngineEvent::Log {
            level: LogLevel::parse(obj.get("level").and_then(Value::as_str).unwrap_or("info")),
            message: text(obj, "message").unwrap_or_default(),
        },
        "progress" => EngineEvent::Progress(read_progress(obj)),
        "metrics" => EngineEvent::Metrics(read_metrics(obj)),
        "artifact" => EngineEvent::Artifact(Artifact {
            kind: text(obj, "kind").unwrap_or_default(),
            path: PathBuf::from(text(obj, "path").unwrap_or_default()),
        }),
        "result" => EngineEvent::Result(read_inference(obj.get("payload"))?),
        "done" => EngineEvent::Done(read_done(obj)),
        "error" => {
            EngineEvent::Error(text(obj, "message").unwrap_or_else(|| "未知错误".to_string()))
        }
        _ => return None,
    })
}

fn read_progress(obj: &Map<String, Value>) -> Progress {
    Progress {
        stage: text(obj, "stage").unwrap_or_default(),
        current: number(obj, "current").unwrap_or(0.0),
        total: number(obj, "total").unwrap_or(0.0),
        epoch: integer(obj, "epoch"),
        epochs: integer(obj, "epochs"),
        percent: number(obj, "percent").map(|value| value as f32),
    }
}

fn read_metrics(obj: &Map<String, Value>) -> Metrics {
    let (values, per_class) = read_values(obj.get("values"));
    Metrics {
        epoch: integer(obj, "epoch").unwrap_or(0),
        epochs: integer(obj, "epochs").unwrap_or(0),
        values,
        per_class,
        is_final: obj.get("final").and_then(Value::as_bool).unwrap_or(false),
    }
}

fn read_inference(payload: Option<&Value>) -> Option<Inference> {
    let obj = payload?.as_object()?;
    let detections = obj
        .get("detections")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(read_detection).collect())
        .unwrap_or_default();
    Some(Inference {
        image: PathBuf::from(text(obj, "image").unwrap_or_default()),
        width: integer(obj, "width").unwrap_or(0),
        height: integer(obj, "height").unwrap_or(0),
        detections,
        speed: read_scalars(obj.get("speed")),
        annotated: text(obj, "annotated")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from),
    })
}

fn read_detection(value: &Value) -> Option<Detection> {
    let obj = value.as_object()?;
    Some(Detection {
        class_id: integer(obj, "class_id").unwrap_or(0),
        name: text(obj, "name").unwrap_or_default(),
        confidence: number(obj, "confidence").unwrap_or(0.0) as f32,
        xyxy: float_array(obj.get("xyxy")).unwrap_or_default(),
        xywhn: float_array(obj.get("xywhn")).unwrap_or_default(),
    })
}

fn read_done(obj: &Map<String, Value>) -> Done {
    let summary = text(obj, "summary").unwrap_or_default();
    let (metrics, per_class) = read_values(obj.get("metrics"));
    let fields = obj
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "type" | "summary" | "metrics" | "values" | "per_class"
            )
        })
        .map(|(key, value)| (key.clone(), format_value(value)))
        .collect();
    Done {
        summary,
        fields,
        metrics,
        per_class,
    }
}

fn read_values(value: Option<&Value>) -> (BTreeMap<String, f64>, Vec<ClassMetric>) {
    let mut metrics = BTreeMap::new();
    let mut per_class = Vec::new();
    let Some(obj) = value.and_then(Value::as_object) else {
        return (metrics, per_class);
    };
    for (key, item) in obj {
        if key == "per_class" {
            per_class = read_per_class(item);
            continue;
        }
        if let Some(number) = scalar(item) {
            metrics.insert(key.clone(), number);
        }
    }
    (metrics, per_class)
}

fn read_per_class(value: &Value) -> Vec<ClassMetric> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            Some(ClassMetric {
                class_id: integer(obj, "class_id").unwrap_or(0),
                name: text(obj, "name").unwrap_or_default(),
                map: number(obj, "mAP50-95")
                    .or_else(|| number(obj, "map"))
                    .unwrap_or(f64::NAN),
            })
        })
        .collect()
}

fn read_scalars(value: Option<&Value>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    if let Some(obj) = value.and_then(Value::as_object) {
        for (key, item) in obj {
            if let Some(number) = scalar(item) {
                out.insert(key.clone(), number);
            }
        }
    }
    out
}

fn scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn text(obj: &Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key).and_then(Value::as_str).map(str::to_string)
}

fn number(obj: &Map<String, Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(scalar)
}

fn integer(obj: &Map<String, Value>, key: &str) -> Option<u32> {
    number(obj, key).map(|value| value.max(0.0).round() as u32)
}

fn float_array(value: Option<&Value>) -> Option<[f32; 4]> {
    let items = value?.as_array()?;
    if items.len() < 4 {
        return None;
    }
    let mut out = [0.0f32; 4];
    for (slot, item) in out.iter_mut().zip(items.iter()) {
        *slot = scalar(item)? as f32;
    }
    Some(out)
}

fn format_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.as_f64().map_or_else(|| number.to_string(), fmt_num),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "-".to_string(),
        other => {
            let text = serde_json::to_string(other).unwrap_or_default();
            if text.chars().count() > 120 {
                let mut short: String = text.chars().take(120).collect();
                short.push('…');
                short
            } else {
                text
            }
        }
    }
}

/// 数字的紧凑显示：整数不带小数点，小数最多保留 4 位。
pub fn fmt_num(value: f64) -> String {
    if !value.is_finite() {
        return "-".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    if !(1e-4..1e6).contains(&abs) {
        return format!("{value:.3e}");
    }
    if (value - value.round()).abs() < 1e-9 {
        return format!("{value:.0}");
    }
    let text = format!("{value:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_log_event() {
        let event = parse_line(r#"{"type":"log","level":"warn","message":"小心"}"#).unwrap();
        match event {
            EngineEvent::Log { level, message } => {
                assert_eq!(level, LogLevel::Warn);
                assert_eq!(message, "小心");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_progress_event() {
        let line =
            r#"{"type":"progress","stage":"epoch","current":0,"total":3,"epoch":1,"epochs":3}"#;
        match parse_line(line).unwrap() {
            EngineEvent::Progress(progress) => {
                assert_eq!(progress.stage, "epoch");
                assert_eq!(progress.epoch, Some(1));
                assert_eq!(progress.text(), "第 1/3 轮");
                assert_eq!(progress.ratio(), 0.0);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_metrics_with_per_class() {
        let line = r#"{"type":"metrics","epoch":2,"epochs":2,"values":{"metrics/mAP50(B)":0.5,"per_class":[{"class_id":3,"name":"V","mAP50-95":0.25}]}}"#;
        match parse_line(line).unwrap() {
            EngineEvent::Metrics(metrics) => {
                assert_eq!(metrics.epoch, 2);
                assert!((metrics.lookup(&["metrics/mAP50"]).unwrap() - 0.5).abs() < 1e-9);
                assert_eq!(metrics.per_class.len(), 1);
                assert_eq!(metrics.per_class[0].name, "V");
                assert!(!metrics.is_final);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_result_event() {
        let line = r#"{"type":"result","payload":{"image":"a.jpg","width":640,"height":480,"detections":[{"class_id":1,"name":"A","confidence":0.9,"xyxy":[1,2,3,4],"xywhn":[0.5,0.5,0.1,0.1]}],"speed":{"inference":4.5},"annotated":""}}"#;
        match parse_line(line).unwrap() {
            EngineEvent::Result(inference) => {
                assert_eq!(inference.file_name(), "a.jpg");
                assert_eq!(inference.detections.len(), 1);
                assert_eq!(inference.detections[0].xyxy, [1.0, 2.0, 3.0, 4.0]);
                assert!(inference.annotated.is_none());
                assert!((inference.inference_ms().unwrap() - 4.5).abs() < 1e-9);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_done_event() {
        let line =
            r#"{"type":"done","summary":"完成","images":3,"detections":7,"metrics":{"mAP50":0.5}}"#;
        match parse_line(line).unwrap() {
            EngineEvent::Done(done) => {
                assert_eq!(done.summary, "完成");
                assert_eq!(done.field("images"), Some("3"));
                assert_eq!(done.field("detections"), Some("7"));
                assert_eq!(done.metrics["mAP50"], 0.5);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn done_metrics_convert_to_metrics() {
        let line = r#"{"type":"done","summary":"完成","metrics":{"mAP50":0.5,
            "per_class":[{"class_id":2,"name":"D","mAP50-95":0.31}]}}"#;
        let EngineEvent::Done(done) = parse_line(line).unwrap() else {
            panic!("应为 done 事件");
        };
        let metrics = done.to_metrics().expect("done 里带指标");
        assert!(metrics.is_final);
        assert_eq!(metrics.get("mAP50"), Some(0.5));
        assert_eq!(metrics.per_class.len(), 1);
        assert_eq!(metrics.per_class[0].class_id, 2);

        let plain = Done {
            summary: "无指标".to_string(),
            ..Default::default()
        };
        assert!(plain.to_metrics().is_none());
    }

    #[test]
    fn ignores_noise() {
        assert!(parse_line("Loading weights...").is_none());
        assert!(parse_line(r#"{"type":"unknown-thing"}"#).is_none());
        assert!(parse_line("{}").is_none());
    }

    #[test]
    fn formats_numbers() {
        assert_eq!(fmt_num(0.0), "0");
        assert_eq!(fmt_num(3.0), "3");
        assert_eq!(fmt_num(0.5), "0.5");
        assert_eq!(fmt_num(-0.123456), "-0.1235");
        assert_eq!(fmt_num(f64::NAN), "-");
    }
}
