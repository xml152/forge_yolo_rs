//! 子进程封装：启动 `python -m engine <子命令>`，把 stdout 的协议行与 stderr 的
//! 普通日志分别读取回主线程，并支持中途取消。

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, unbounded};

use super::protocol::{self, EngineEvent};

/// 子进程回传的一帧数据。
pub enum EngineMessage {
    /// stdout 上的一条协议事件。
    Event(EngineEvent),
    /// stderr 上的一行普通日志（ultralytics 输出、tqdm、警告……）。
    Stderr(String),
}

/// 任务结束信息。
#[derive(Clone, Copy, Debug)]
pub struct TaskOutcome {
    pub exit_code: Option<i32>,
    pub success: bool,
}

impl TaskOutcome {
    pub fn describe(self) -> String {
        match self.exit_code {
            Some(0) => "进程正常退出".to_string(),
            Some(code) => format!("进程退出，返回码 {code}"),
            None => "进程被信号终止".to_string(),
        }
    }
}

/// 一个运行中的 Python 引擎任务。
pub struct EngineHandle {
    rx: Receiver<EngineMessage>,
    child: Child,
    readers: Vec<JoinHandle<()>>,
    pending_readers: Arc<AtomicUsize>,
    exit_code: Option<Option<i32>>,
    finished: bool,
}

impl EngineHandle {
    /// 启动子进程。
    ///
    /// `python` 是解释器路径，`engine_root` 作为工作目录（引擎内部大量使用
    /// 相对路径，如 `weights/yolo26n.pt`、`runs/...`），`args` 为子命令及参数。
    pub fn spawn(python: &str, engine_root: &Path, args: &[String]) -> Result<Self> {
        if !engine_root.is_dir() {
            bail!("找不到引擎目录：{}", engine_root.display());
        }

        let mut command = Command::new(python);
        command
            .arg("-m")
            .arg("engine")
            .args(args)
            .current_dir(engine_root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("TERM", "dumb")
            .env("NO_COLOR", "1");

        // 独立进程组，便于取消时连同 dataloader 子进程一起结束。
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let mut child = command
            .spawn()
            .with_context(|| format!("无法启动 Python 引擎：{python}"))?;

        let stdout = child.stdout.take().context("无法读取引擎 stdout")?;
        let stderr = child.stderr.take().context("无法读取引擎 stderr")?;

        let (tx, rx) = unbounded();
        let pending = Arc::new(AtomicUsize::new(2));
        let readers = vec![
            spawn_reader(stdout, tx.clone(), pending.clone(), true),
            spawn_reader(stderr, tx, pending.clone(), false),
        ];

        Ok(Self {
            rx,
            child,
            readers,
            pending_readers: pending,
            exit_code: None,
            finished: false,
        })
    }

    /// 取出已到达的消息。
    pub fn drain(&self) -> Vec<EngineMessage> {
        self.rx.try_iter().collect()
    }

    /// 刷新进程状态；首次检测到退出时返回结束信息。
    pub fn poll(&mut self) -> Option<TaskOutcome> {
        if self.finished {
            return None;
        }
        if self.exit_code.is_none() {
            match self.child.try_wait() {
                Ok(Some(status)) => self.exit_code = Some(status.code()),
                Ok(None) => return None,
                Err(_) => self.exit_code = Some(None),
            }
        }
        // 子进程退出后仍需等读取线程把管道里剩余的行消费完。
        if self.pending_readers.load(Ordering::SeqCst) > 0 {
            return None;
        }
        self.finished = true;
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
        let code = self.exit_code.flatten();
        Some(TaskOutcome {
            exit_code: code,
            success: code == Some(0),
        })
    }

    /// 请求结束进程（连同其子进程组）。
    pub fn cancel(&mut self) {
        if self.finished {
            return;
        }
        #[cfg(unix)]
        {
            let pid = self.child.id() as i32;
            // 负号表示整个进程组；忽略返回值，后面还有兜底的 kill。
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
        }
        let _ = self.child.kill();
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        self.cancel();
        let _ = self.child.wait();
    }
}

fn spawn_reader<R: std::io::Read + Send + 'static>(
    stream: R,
    tx: Sender<EngineMessage>,
    pending: Arc<AtomicUsize>,
    protocol: bool,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines() {
            let Ok(line) = line else { break };
            let line = line.trim_end().to_string();
            if line.is_empty() {
                continue;
            }
            let message = if protocol {
                match protocol::parse_line(&line) {
                    Some(event) => EngineMessage::Event(event),
                    // stdout 上出现非协议内容时，当作日志显示而不是丢弃。
                    None => EngineMessage::Stderr(line),
                }
            } else {
                EngineMessage::Stderr(line)
            };
            if tx.send(message).is_err() {
                break;
            }
        }
        pending.fetch_sub(1, Ordering::SeqCst);
    })
}
