//! Rust 前端与 Python 训练引擎之间的桥接层。
//!
//! 引擎以 `python -m engine <子命令>` 的形式作为子进程运行，stdout 上只输出
//! JSON Lines 协议事件（见 [`protocol`]），stderr 上的普通日志也被收集起来。
//! [`Job`] 把这条流水线封装成一个可在 GUI 里逐帧轮询的任务对象。

pub mod process;
pub mod protocol;

use std::path::{Path, PathBuf};

pub use process::{EngineHandle, EngineMessage, TaskOutcome};
pub use protocol::{ClassMetric, Detection, EngineEvent, Inference, LogLevel, Metrics, Progress};

/// 引擎目录的候选位置：环境变量 > 当前目录/engine > 可执行文件旁边的 engine。
pub fn locate_engine_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("FORGE_ENGINE_ROOT") {
        let root = PathBuf::from(root);
        if is_engine_root(&root) {
            return Some(root);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("engine");
        if is_engine_root(&candidate) {
            return Some(candidate);
        }
        if is_engine_root(&cwd) {
            return Some(cwd);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1).take(4) {
            let candidate = ancestor.join("engine");
            if is_engine_root(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn is_engine_root(path: &Path) -> bool {
    path.join("src/engine/__init__.py").is_file()
}

/// 自动挑选解释器：优先用引擎自带的虚拟环境，其次系统 python3/python。
pub fn discover_python(engine_root: &Path) -> String {
    let venv = engine_root.join(".venv/bin/python");
    if venv.is_file() {
        return venv.display().to_string();
    }
    for name in ["python3", "python"] {
        if which(name) {
            return name.to_string();
        }
    }
    "python3".to_string()
}

fn which(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(name);
        candidate.is_file()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Idle,
    Running,
    Finished,
    Failed,
}

/// 一行日志及其来源级别。
#[derive(Clone, Debug)]
pub struct LogLine {
    pub level: LogLevel,
    pub text: String,
}

/// 一个正在运行或已结束的引擎任务。
pub struct Job {
    kind: &'static str,
    handle: Option<EngineHandle>,
    pub status: JobStatus,
    pub logs: Vec<LogLine>,
    pub progress: Option<Progress>,
    pub error: Option<String>,
    pub outcome: Option<TaskOutcome>,
}

/// `Job::pump` 的结果：除日志以外的协议事件，交给页面自行解释。
pub struct JobUpdate {
    pub events: Vec<EngineEvent>,
    pub finished: bool,
}

impl Job {
    pub fn new(kind: &'static str) -> Self {
        Self {
            kind,
            handle: None,
            status: JobStatus::Idle,
            logs: Vec::new(),
            progress: None,
            error: None,
            outcome: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.status == JobStatus::Running
    }

    /// 任务类别（`train` / `infer` / `val` / `export`），用于日志标题。
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// 日志标题里用的 `类别 · 行数` 文本。
    pub fn title(&self) -> String {
        format!("{} · {} 行", self.kind, self.logs.len())
    }

    pub fn start(
        &mut self,
        python: &str,
        engine_root: &Path,
        args: &[String],
    ) -> anyhow::Result<()> {
        self.stop();
        self.logs.clear();
        self.progress = None;
        self.error = None;
        self.outcome = None;
        let handle = EngineHandle::spawn(python, engine_root, args)?;
        let command = format!("{python} -m engine {}", args.join(" "));
        self.logs.push(LogLine {
            level: LogLevel::Info,
            text: format!("$ {command}"),
        });
        self.handle = Some(handle);
        self.status = JobStatus::Running;
        Ok(())
    }

    /// 拉取新消息。日志与进度在内部消费，其余事件原样返回。
    pub fn pump(&mut self) -> JobUpdate {
        let mut events = Vec::new();
        let mut finished = false;
        let Some(handle) = self.handle.as_mut() else {
            return JobUpdate { events, finished };
        };

        for message in handle.drain() {
            match message {
                EngineMessage::Stderr(text) => self.logs.push(LogLine {
                    level: LogLevel::Info,
                    text,
                }),
                EngineMessage::Event(EngineEvent::Log { level, message }) => {
                    self.logs.push(LogLine {
                        level,
                        text: message,
                    });
                }
                EngineMessage::Event(EngineEvent::Progress(progress)) => {
                    self.progress = Some(progress);
                }
                EngineMessage::Event(EngineEvent::Error(message)) => {
                    self.error = Some(message.clone());
                    self.logs.push(LogLine {
                        level: LogLevel::Error,
                        text: message.clone(),
                    });
                    events.push(EngineEvent::Error(message));
                }
                EngineMessage::Event(event) => events.push(event),
            }
        }

        if let Some(outcome) = handle.poll() {
            self.outcome = Some(outcome);
            self.status = if outcome.success && self.error.is_none() {
                JobStatus::Finished
            } else {
                JobStatus::Failed
            };
            if self.status == JobStatus::Failed && self.error.is_none() {
                self.error = Some(outcome.describe());
            }
            self.handle = None;
            finished = true;
        }

        if self.logs.len() > MAX_LOG_LINES {
            let overflow = self.logs.len() - MAX_LOG_LINES;
            self.logs.drain(..overflow);
        }

        JobUpdate { events, finished }
    }

    pub fn cancel(&mut self) {
        if let Some(handle) = self.handle.as_mut() {
            handle.cancel();
            self.logs.push(LogLine {
                level: LogLevel::Warn,
                text: "已请求终止任务".to_string(),
            });
        }
    }

    pub fn stop(&mut self) {
        if let Some(mut handle) = self.handle.take() {
            handle.cancel();
            let _ = handle.poll();
            if self.status == JobStatus::Running {
                self.status = JobStatus::Idle;
            }
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.stop();
    }
}

const MAX_LOG_LINES: usize = 5000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_root_detection_finds_repo_engine() {
        // 单测在 crate 根目录下运行，能定位到仓库里的 engine/。
        let root = locate_engine_root().expect("应能找到 engine 目录");
        assert!(is_engine_root(&root));
        assert!(!discover_python(&root).is_empty());
    }
}

/// 真跑一次 Python 引擎，验证 spawn → 协议解析 → Job 状态机整条链路。
/// 需要 engine 环境就绪，故标记 `#[ignore]`，用 `cargo test -- --ignored` 手动执行。
#[cfg(test)]
mod e2e_tests {
    use super::*;

    #[test]
    #[ignore = "需要 engine 虚拟环境，手动执行：cargo test -- --ignored"]
    fn runs_val_end_to_end() {
        let root = locate_engine_root().expect("应能找到 engine 目录");
        let python = discover_python(&root);
        let data = root.join("datasets/hands_gesture/data.yaml");
        let weights = root.join("weights/yolo26n.pt");

        let mut job = Job::new("val");
        job.start(
            &python,
            &root,
            &[
                "val".to_string(),
                "--weights".to_string(),
                weights.display().to_string(),
                "--data".to_string(),
                data.display().to_string(),
                "--imgsz".to_string(),
                "320".to_string(),
                "--batch".to_string(),
                "8".to_string(),
            ],
        )
        .expect("启动失败");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        let mut done = None;
        let mut saw_event = false;
        while std::time::Instant::now() < deadline {
            let update = job.pump();
            for event in update.events {
                saw_event = true;
                if let EngineEvent::Done(payload) = event {
                    done = Some(payload);
                }
            }
            if update.finished {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        assert_eq!(job.status, JobStatus::Finished, "错误：{:?}", job.error);
        assert!(saw_event, "未收到任何协议事件");
        assert!(!job.logs.is_empty(), "未收集到 stderr 日志");
        let done = done.expect("未收到 done 事件");
        let metrics = done.to_metrics().expect("done 里应带指标");
        assert!(metrics.get("precision").is_some(), "缺少 precision");
        assert!(metrics.get("mAP50-95").is_some(), "缺少 mAP50-95");
    }
}

#[cfg(test)]
mod e2e_infer_tests {
    use super::*;

    #[test]
    #[ignore = "需要 engine 虚拟环境，手动执行：cargo test -- --ignored"]
    fn runs_infer_end_to_end() {
        let root = locate_engine_root().expect("应能找到 engine 目录");
        let python = discover_python(&root);
        let weights = root.join("weights/yolo26n.pt");
        let source = root.join("datasets/hands_gesture/images/val");

        let mut job = Job::new("infer");
        job.start(
            &python,
            &root,
            &[
                "infer".to_string(),
                "--weights".to_string(),
                weights.display().to_string(),
                "--source".to_string(),
                source.display().to_string(),
                "--imgsz".to_string(),
                "320".to_string(),
                "--no-save".to_string(),
            ],
        )
        .expect("启动失败");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        let mut images = 0usize;
        let mut detections = 0usize;
        let mut done = None;
        while std::time::Instant::now() < deadline {
            let update = job.pump();
            for event in update.events {
                match event {
                    EngineEvent::Result(result) => {
                        images += 1;
                        detections += result.detections.len();
                        assert!(result.width > 0 && result.height > 0, "图片尺寸异常");
                        for det in &result.detections {
                            let [x1, y1, x2, y2] = det.xyxy;
                            assert!(x1 <= x2 && y1 <= y2, "坐标顺序错误");
                            let norm = det.xywhn;
                            assert!(norm.iter().all(|v| (0.0..=1.0).contains(v)));
                        }
                    }
                    EngineEvent::Done(payload) => done = Some(payload),
                    _ => {}
                }
            }
            if update.finished {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        assert_eq!(job.status, JobStatus::Finished, "错误：{:?}", job.error);
        assert_eq!(images, 200, "应收到 200 张图的结果");
        assert!(detections > 0, "总目标数为 0");
        let done = done.expect("未收到 done 事件");
        assert_eq!(done.field("images"), Some("200"));
    }
}

#[cfg(test)]
mod e2e_export_tests {
    use super::*;

    #[test]
    #[ignore = "需要 engine 虚拟环境，手动执行：cargo test -- --ignored"]
    fn runs_export_end_to_end() {
        let root = locate_engine_root().expect("应能找到 engine 目录");
        let python = discover_python(&root);
        let weights = root.join("weights/yolo26n.pt");

        let mut job = Job::new("export");
        job.start(
            &python,
            &root,
            &[
                "export".to_string(),
                "--weights".to_string(),
                weights.display().to_string(),
                "--format".to_string(),
                "onnx".to_string(),
                "--imgsz".to_string(),
                "160".to_string(),
            ],
        )
        .expect("启动失败");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        let mut artifact = None;
        let mut done = None;
        while std::time::Instant::now() < deadline {
            let update = job.pump();
            for event in update.events {
                match event {
                    EngineEvent::Artifact(a) => artifact = Some(a),
                    EngineEvent::Done(payload) => done = Some(payload),
                    _ => {}
                }
            }
            if update.finished {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        assert_eq!(job.status, JobStatus::Finished, "错误：{:?}", job.error);
        let artifact = artifact.expect("未收到 artifact 事件");
        assert!(
            artifact.path.is_file(),
            "导出文件不存在：{:?}",
            artifact.path
        );
        assert_eq!(
            artifact.path.extension().and_then(|e| e.to_str()),
            Some("onnx")
        );
        let done = done.expect("未收到 done 事件");
        assert_eq!(done.field("format"), Some("onnx"));

        let _ = std::fs::remove_file(&artifact.path);
    }
}

#[cfg(test)]
mod e2e_train_tests {
    use super::*;

    #[test]
    #[ignore = "需要 engine 虚拟环境与 GPU，手动执行：cargo test -- --ignored"]
    fn runs_train_end_to_end() {
        let root = locate_engine_root().expect("应能找到 engine 目录");
        let python = discover_python(&root);
        let weights = root.join("weights/yolo26n.pt");
        let data = root.join("datasets/hands_gesture/data.yaml");
        let project = root.join("runs/e2e-train");

        let mut job = Job::new("train");
        job.start(
            &python,
            &root,
            &[
                "train".to_string(),
                "--weights".to_string(),
                weights.display().to_string(),
                "--data".to_string(),
                data.display().to_string(),
                "--epochs".to_string(),
                "1".to_string(),
                "--imgsz".to_string(),
                "160".to_string(),
                "--batch".to_string(),
                "16".to_string(),
                "--workers".to_string(),
                "4".to_string(),
                "--project".to_string(),
                project.display().to_string(),
                "--name".to_string(),
                "e2e".to_string(),
            ],
        )
        .expect("启动失败");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(900);
        let mut rounds = Vec::new();
        let mut finals = Vec::new();
        let mut best: Option<PathBuf> = None;
        let mut done = None;
        while std::time::Instant::now() < deadline {
            let update = job.pump();
            for event in update.events {
                match event {
                    EngineEvent::Metrics(metrics) => {
                        if metrics.is_final {
                            finals.push(metrics);
                        } else {
                            rounds.push(metrics);
                        }
                    }
                    EngineEvent::Artifact(artifact) if artifact.kind == "best" => {
                        best = Some(artifact.path);
                    }
                    EngineEvent::Done(payload) => done = Some(payload),
                    _ => {}
                }
            }
            if update.finished {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        assert_eq!(job.status, JobStatus::Finished, "错误：{:?}", job.error);
        assert_eq!(rounds.len(), 1, "单轮训练应只上报 1 条普通 metrics");
        assert_eq!(
            finals.len(),
            1,
            "复评应只上报 1 条 final metrics，实际 {} 条",
            finals.len()
        );

        let final_metrics = &finals[0];
        // 引擎上报的是 ultralytics 原始键 "metrics/xxx(B)"，UI 用裸键兜底。
        let cases: [(&str, &str); 4] = [
            ("precision", "metrics/precision(B)"),
            ("recall", "metrics/recall(B)"),
            ("mAP50", "metrics/mAP50(B)"),
            ("mAP50-95", "metrics/mAP50-95(B)"),
        ];
        for (bare, prefixed) in cases {
            let found = final_metrics.lookup(&[bare, prefixed]);
            assert!(found.is_some(), "final metrics 缺少 {bare}");
        }
        assert!(
            !final_metrics.per_class.is_empty(),
            "final metrics 应带逐类别指标"
        );
        assert!(
            !rounds[0].values.is_empty(),
            "普通 metrics 应带 loss 等数值"
        );

        let best = best.expect("未收到 best artifact");
        assert!(best.is_file(), "best.pt 不存在：{best:?}");
        done.expect("未收到 done 事件");

        let _ = std::fs::remove_dir_all(&project);
    }
}
