//! 模型训练页面：参数表单 + Python 子进程任务 + 实时指标、产物与日志。

use std::path::Path;

use egui::{Color32, RichText, Ui};

use crate::bridge::{self, ClassMetric, JobStatus, protocol::fmt_num};
use crate::state::{AppState, CacheMode, EXPORT_FORMATS};
use crate::ui::class_color;

pub fn show(ui: &mut Ui, state: &mut AppState) {
    // 先把子进程这一帧产生的输出消费掉。
    state.train.pump();
    state.train.export.pump();

    egui::Panel::bottom("train_log_panel")
        .resizable(true)
        .default_size(190.0)
        .show(ui, |ui| log_panel(ui, state));
    egui::Panel::right("train_side_panel")
        .resizable(true)
        .default_size(330.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("train_side_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    metrics_panel(ui, state);
                    ui.separator();
                    artifacts_panel(ui, state);
                });
        });
    egui::CentralPanel::default().show(ui, |ui| {
        egui::ScrollArea::vertical()
            .id_salt("train_form_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                train_form(ui, state);
                ui.add_space(12.0);
                export_form(ui, state);
            });
    });
}

fn train_form(ui: &mut Ui, state: &mut AppState) {
    ui.heading("模型训练");
    ui.label("底层由 ultralytics 训练，Rust 侧负责收集参数、驱动子进程并展示进度。");
    ui.add_space(6.0);

    let running = state.train.job.is_running();

    ui.add_enabled_ui(!running, |ui| {
        egui::Grid::new("train_form_grid")
            .num_columns(2)
            .spacing([10.0, 6.0])
            .show(ui, |ui| {
                ui.label("数据集目录：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.dataset_root)
                            .desired_width(300.0)
                            .hint_text("datasets/hands_gesture"),
                    );
                    if ui.button("浏览…").clicked() {
                        if let Some(dir) = rfd::FileDialog::new()
                            .set_title("选择数据集根目录")
                            .pick_folder()
                        {
                            state.train.dataset_root = dir.display().to_string();
                        }
                    }
                    if ui.button("用标注页的数据集").clicked() {
                        if let Some(dataset) = &state.annotator.dataset {
                            state.train.dataset_root = dataset.root.display().to_string();
                        }
                    }
                });
                ui.end_row();

                ui.label("data.yaml：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.data_yaml)
                            .desired_width(300.0)
                            .hint_text("留空则按数据集目录自动生成"),
                    );
                    if ui.button("选择…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("YAML", &["yaml", "yml"])
                            .pick_file()
                        {
                            state.train.data_yaml = path.display().to_string();
                        }
                    }
                });
                ui.end_row();

                ui.label("初始权重：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.weights)
                            .desired_width(300.0)
                            .hint_text("留空则用 engine/weights/yolo26n.pt"),
                    );
                    if ui.button("选择…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("PyTorch 权重", &["pt"])
                            .pick_file()
                        {
                            state.train.weights = path.display().to_string();
                        }
                    }
                });
                ui.end_row();

                ui.label("训练轮次：");
                ui.add(
                    egui::DragValue::new(&mut state.train.epochs)
                        .speed(1.0)
                        .range(1..=1000),
                );
                ui.end_row();

                ui.label("图像尺寸：");
                ui.add(
                    egui::DragValue::new(&mut state.train.imgsz)
                        .speed(32.0)
                        .range(64..=2048),
                );
                ui.end_row();

                ui.label("批大小：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut state.train.batch)
                            .speed(1.0)
                            .range(-1..=512),
                    );
                    ui.weak("-1 表示自动估算");
                });
                ui.end_row();

                ui.label("数据加载进程：");
                ui.add(
                    egui::DragValue::new(&mut state.train.workers)
                        .speed(1.0)
                        .range(0..=32),
                );
                ui.end_row();

                ui.label("设备：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.device)
                            .desired_width(120.0)
                            .hint_text("0 / 0,1 / cpu"),
                    );
                    ui.weak("留空自动选择");
                });
                ui.end_row();

                ui.label("优化器：");
                egui::ComboBox::from_id_salt("train_optimizer")
                    .width(140.0)
                    .selected_text(state.train.optimizer.clone())
                    .show_ui(ui, |ui| {
                        for name in ["auto", "SGD", "Adam", "AdamW"] {
                            ui.selectable_value(&mut state.train.optimizer, name.to_string(), name);
                        }
                    });
                ui.end_row();

                ui.label("初始学习率：");
                ui.add(
                    egui::DragValue::new(&mut state.train.lr0)
                        .speed(0.0005)
                        .range(0.0..=1.0),
                );
                ui.end_row();

                ui.label("早停耐心值：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut state.train.patience)
                            .speed(1.0)
                            .range(0..=1000),
                    );
                    ui.weak("0 表示关闭早停");
                });
                ui.end_row();

                ui.label("图像缓存：");
                egui::ComboBox::from_id_salt("train_cache")
                    .width(140.0)
                    .selected_text(state.train.cache.label())
                    .show_ui(ui, |ui| {
                        for mode in CacheMode::ALL {
                            ui.selectable_value(&mut state.train.cache, mode, mode.label());
                        }
                    });
                ui.end_row();

                ui.label("输出目录：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.project)
                            .desired_width(180.0)
                            .hint_text("runs/detect"),
                    );
                    ui.label("/");
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.name)
                            .desired_width(110.0)
                            .hint_text("train"),
                    );
                });
                ui.end_row();

                ui.label("其他：");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut state.train.amp, "混合精度 AMP");
                    ui.checkbox(&mut state.train.plots, "保存训练曲线图");
                });
                ui.end_row();
            });
    });

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!running, egui::Button::new("开始训练"))
            .clicked()
        {
            if let Some(root) = bridge::locate_engine_root() {
                let python = state.resolve_python(&root);
                if let Err(err) = state.train.start(&python, &root) {
                    state.train.status = format!("{err:#}");
                }
            } else {
                state.train.status = "未找到 engine 目录，请设置 FORGE_ENGINE_ROOT".to_string();
            }
        }
        if ui.add_enabled(running, egui::Button::new("取消")).clicked() {
            state.train.job.cancel();
            state.train.status = "正在终止训练…".to_string();
        }
        if ui.button("打开输出目录").clicked() {
            let dir = state.train.output_dir();
            if let Err(err) = open_dir(&dir) {
                state.train.status = format!("无法打开目录：{err}");
            }
        }
        ui.separator();
        status_chip(ui, state.train.job.status);
    });

    if !state.train.status.is_empty() {
        ui.label(&state.train.status);
    }

    ui.add_space(6.0);
    progress_block(ui, &state.train.job);
}

fn status_chip(ui: &mut Ui, status: JobStatus) {
    match status {
        JobStatus::Idle => ui.weak("空闲"),
        JobStatus::Running => ui.colored_label(Color32::from_rgb(90, 170, 240), "运行中"),
        JobStatus::Finished => ui.colored_label(Color32::from_rgb(110, 190, 110), "已完成"),
        JobStatus::Failed => ui.colored_label(Color32::from_rgb(230, 120, 110), "失败"),
    };
}

/// 阶段进度条 + 批次进度条。
fn progress_block(ui: &mut Ui, job: &bridge::Job) {
    let Some(progress) = &job.progress else {
        ui.add(
            egui::ProgressBar::new(0.0)
                .desired_height(16.0)
                .text("等待开始"),
        );
        return;
    };
    ui.add(
        egui::ProgressBar::new(progress.ratio())
            .desired_height(16.0)
            .text(format!(
                "{} · {}",
                stage_label(&progress.stage),
                progress.text()
            ))
            .animate(job.is_running()),
    );
    if let Some(percent) = progress.percent {
        ui.add(
            egui::ProgressBar::new(percent.clamp(0.0, 1.0))
                .desired_height(10.0)
                .fill(Color32::from_rgb(120, 170, 110)),
        );
    }
}

/// 把引擎上报的阶段名翻成中文。
pub fn stage_label(stage: &str) -> &str {
    match stage {
        "epoch" => "训练轮次",
        "batch" => "批次",
        "predict" => "推理",
        "val" => "评估",
        "export" => "导出",
        other if other.is_empty() => "进度",
        other => other,
    }
}

fn metrics_panel(ui: &mut Ui, state: &mut AppState) {
    ui.label(RichText::new("训练指标").strong());
    ui.separator();

    let history_len = state.train.history.len();
    let latest = state.train.latest.clone();
    let Some(latest) = latest else {
        ui.weak("尚无指标上报。");
        return;
    };

    ui.label(format!("第 {} / {} 轮", latest.epoch, latest.epochs));
    if latest.is_final {
        ui.colored_label(Color32::from_rgb(110, 190, 110), "最终复评（best.pt）");
    }
    ui.add_space(4.0);

    // 常用指标置顶，其余折叠在后面。ultralytics 训练指标键带 `metrics/…(B)` 形式，
    // 而评估子命令上报的是裸键，两种都试。
    let highlights: [(&str, &[&str]); 5] = [
        ("mAP50-95", &["mAP50-95", "metrics/mAP50-95(B)"]),
        ("mAP50", &["mAP50", "metrics/mAP50(B)"]),
        ("precision", &["precision", "metrics/precision(B)"]),
        ("recall", &["recall", "metrics/recall(B)"]),
        ("box_loss", &["train/box_loss", "val/box_loss"]),
    ];
    egui::Grid::new("train_highlight_grid")
        .num_columns(2)
        .striped(true)
        .spacing([12.0, 3.0])
        .show(ui, |ui| {
            for (label, prefixes) in highlights {
                let Some(value) = latest.lookup(prefixes) else {
                    continue;
                };
                ui.label(label);
                ui.monospace(fmt_num(value));
                ui.end_row();
            }
        });

    ui.add_space(2.0);
    egui::CollapsingHeader::new("全部指标")
        .id_salt("train_all_metrics")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("train_metric_grid")
                .num_columns(2)
                .striped(true)
                .spacing([12.0, 3.0])
                .show(ui, |ui| {
                    for (key, value) in &latest.values {
                        ui.monospace(key);
                        ui.monospace(fmt_num(*value));
                        ui.end_row();
                    }
                });
        });

    if !latest.per_class.is_empty() {
        egui::CollapsingHeader::new("各类别 mAP50-95")
            .id_salt("train_per_class")
            .default_open(true)
            .show(ui, |ui| {
                per_class_grid(ui, "train_per_class_grid", &latest.per_class)
            });
    }

    if history_len > 0 {
        egui::CollapsingHeader::new(format!("逐轮记录（{history_len} 轮）"))
            .id_salt("train_history")
            .default_open(false)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("train_history_scroll")
                    .max_height(220.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Grid::new("train_history_grid")
                            .num_columns(3)
                            .striped(true)
                            .spacing([10.0, 3.0])
                            .show(ui, |ui| {
                                ui.label(RichText::new("轮次").strong());
                                ui.label(RichText::new("mAP50-95").strong());
                                ui.label(RichText::new("box_loss").strong());
                                ui.end_row();
                                for entry in &state.train.history {
                                    ui.monospace(entry.epoch.to_string());
                                    ui.monospace(metric_text(entry, &["metrics/mAP50-95(B)"]));
                                    ui.monospace(metric_text(
                                        entry,
                                        &["train/box_loss", "val/box_loss"],
                                    ));
                                    ui.end_row();
                                }
                            });
                    });
            });
    }
}

fn metric_text(metrics: &bridge::Metrics, prefixes: &[&str]) -> String {
    metrics
        .lookup(prefixes)
        .map(fmt_num)
        .unwrap_or_else(|| "-".to_string())
}

fn per_class_grid(ui: &mut Ui, id: &str, classes: &[ClassMetric]) {
    egui::Grid::new(id)
        .num_columns(3)
        .striped(true)
        .spacing([12.0, 3.0])
        .show(ui, |ui| {
            for metric in classes {
                ui.weak(metric.class_id.to_string());
                ui.label(RichText::new(&metric.name).color(class_color(metric.class_id)));
                ui.monospace(fmt_num(metric.map));
                ui.end_row();
            }
        });
}

fn artifacts_panel(ui: &mut Ui, state: &mut AppState) {
    ui.label(RichText::new("训练产物").strong());
    ui.separator();

    let best = state.train.best.clone();
    let last = state.train.last.clone();

    if best.is_none() && last.is_none() && state.train.summary.is_empty() {
        ui.weak("训练结束后这里会显示权重路径与摘要。");
    }

    if let Some(best) = &best {
        ui.label("best.pt：");
        ui.horizontal_wrapped(|ui| ui.monospace(best.display().to_string()));
    }
    if let Some(last) = &last {
        ui.label("last.pt：");
        ui.horizontal_wrapped(|ui| ui.monospace(last.display().to_string()));
    }
    if !state.train.summary.is_empty() {
        ui.add_space(4.0);
        ui.label(&state.train.summary);
    }
    if let Some(err) = state.train.job.error.clone() {
        ui.add_space(4.0);
        ui.colored_label(Color32::from_rgb(230, 120, 110), err);
    }

    if best.is_some() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.small_button("打开运行目录").clicked() {
                let dir = state.train.output_dir();
                if let Err(err) = open_dir(&dir) {
                    state.train.status = format!("无法打开目录：{err}");
                }
            }
            if ui.small_button("复制 best.pt 路径").clicked() {
                if let Some(best) = &state.train.best {
                    ui.ctx().copy_text(best.display().to_string());
                }
            }
        });
    }
}

fn log_panel(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("运行日志").strong());
        ui.separator();
        ui.weak(state.train.job.title());
        if ui.small_button("清空").clicked() {
            state.train.job.logs.clear();
        }
    });
    ui.separator();
    let default_color = ui.visuals().text_color();
    egui::ScrollArea::vertical()
        .id_salt("train_log_scroll")
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if state.train.job.logs.is_empty() {
                ui.weak("暂无日志。");
            }
            for line in &state.train.job.logs {
                let color = match line.level {
                    bridge::LogLevel::Info => default_color,
                    bridge::LogLevel::Warn => Color32::from_rgb(240, 180, 60),
                    bridge::LogLevel::Error => Color32::from_rgb(230, 120, 110),
                };
                ui.monospace(RichText::new(&line.text).color(color));
            }
        });
}

fn export_form(ui: &mut Ui, state: &mut AppState) {
    ui.separator();
    ui.label(RichText::new("导出模型").strong());
    ui.label("把训练得到的 best.pt 导出成 ONNX 等部署格式。");

    let running = state.train.export.job.is_running();
    ui.add_enabled_ui(!running, |ui| {
        egui::Grid::new("export_grid")
            .num_columns(2)
            .spacing([10.0, 6.0])
            .show(ui, |ui| {
                ui.label("权重：");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut state.train.export.weights)
                            .desired_width(300.0)
                            .hint_text("留空则使用训练得到的 best.pt"),
                    );
                    if ui.button("选择…").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("PyTorch 权重", &["pt"])
                            .pick_file()
                        {
                            state.train.export.weights = path.display().to_string();
                        }
                    }
                    if ui.button("用 best.pt").clicked() {
                        if let Some(best) = &state.train.best {
                            state.train.export.weights = best.display().to_string();
                        }
                    }
                });
                ui.end_row();

                ui.label("目标格式：");
                egui::ComboBox::from_id_salt("export_format")
                    .width(160.0)
                    .selected_text(state.train.export.format.clone())
                    .show_ui(ui, |ui| {
                        for format in EXPORT_FORMATS {
                            ui.selectable_value(
                                &mut state.train.export.format,
                                (*format).to_string(),
                                *format,
                            );
                        }
                    });
                ui.end_row();

                ui.label("输入尺寸：");
                ui.add(
                    egui::DragValue::new(&mut state.train.export.imgsz)
                        .speed(32.0)
                        .range(64..=2048),
                );
                ui.end_row();

                ui.label("ONNX opset：");
                ui.add(
                    egui::DragValue::new(&mut state.train.export.opset)
                        .speed(1.0)
                        .range(9..=21),
                );
                ui.end_row();

                ui.label("选项：");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut state.train.export.half, "FP16");
                    ui.checkbox(&mut state.train.export.dynamic, "动态尺寸");
                    ui.checkbox(&mut state.train.export.simplify, "简化计算图");
                });
                ui.end_row();
            });
    });

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!running, egui::Button::new("导出"))
            .clicked()
        {
            if let Some(root) = bridge::locate_engine_root() {
                let python = state.resolve_python(&root);
                let best = state.train.best.clone();
                if let Err(err) = state.train.export.start(&python, &root, best.as_deref()) {
                    state.train.export.status = format!("{err:#}");
                }
            } else {
                state.train.export.status =
                    "未找到 engine 目录，请设置 FORGE_ENGINE_ROOT".to_string();
            }
        }
        if ui
            .add_enabled(running, egui::Button::new("取消导出"))
            .clicked()
        {
            state.train.export.job.cancel();
        }
        if ui.button("打开所在目录").clicked() {
            let dir = state
                .train
                .export
                .output
                .as_deref()
                .and_then(Path::parent)
                .map(Path::to_path_buf)
                .unwrap_or_else(|| state.train.output_dir());
            if let Err(err) = open_dir(&dir) {
                state.train.export.status = format!("无法打开目录：{err}");
            }
        }
        ui.separator();
        status_chip(ui, state.train.export.job.status);
    });

    if running {
        if let Some(progress) = &state.train.export.job.progress {
            ui.add(
                egui::ProgressBar::new(progress.ratio())
                    .desired_height(12.0)
                    .animate(true),
            );
        }
    }

    if let Some(output) = &state.train.export.output {
        ui.horizontal_wrapped(|ui| {
            ui.label("输出：");
            ui.monospace(output.display().to_string());
        });
    }
    if !state.train.export.status.is_empty() {
        ui.label(&state.train.export.status);
    }
}

/// 用系统文件管理器打开目录。
pub fn open_dir(path: &Path) -> std::io::Result<()> {
    if !path.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} 不存在", path.display()),
        ));
    }
    let command = if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    std::process::Command::new(command)
        .arg(path)
        .spawn()
        .map(|_| ())
}
