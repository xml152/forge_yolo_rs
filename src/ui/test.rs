//! 模型测试页面：图片推理预览 + 验证集指标评估。

use std::path::Path;

use egui::{Align2, Color32, FontId, Rect, RichText, Sense, Stroke, StrokeKind, Ui, pos2};

use crate::bridge::{self, Detection, Inference, JobStatus, protocol::fmt_num};
use crate::state::{AppState, VAL_SPLITS};

pub fn show(ui: &mut Ui, state: &mut AppState) {
    state.test.pump();
    state.test.pump_val();

    egui::Panel::top("test_infer_form")
        .resizable(false)
        .show(ui, |ui| {
            ui.add_space(2.0);
            infer_form(ui, state);
            ui.add_space(2.0);
        });
    egui::Panel::right("test_val_panel")
        .resizable(true)
        .default_size(320.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("test_val_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    val_form(ui, state);
                    ui.separator();
                    val_metrics(ui, state);
                });
        });
    egui::Panel::bottom("test_log_panel")
        .resizable(true)
        .default_size(120.0)
        .show(ui, |ui| log_panel(ui, state));
    egui::CentralPanel::default().show(ui, |ui| {
        egui::Panel::left("test_result_list")
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| result_list(ui, state));
        egui::CentralPanel::default().show(ui, |ui| preview(ui, state));
    });
}

fn infer_form(ui: &mut Ui, state: &mut AppState) {
    let running = state.test.job.is_running();

    ui.horizontal(|ui| {
        ui.label("权重：");
        ui.add(
            egui::TextEdit::singleline(&mut state.test.weights)
                .desired_width(280.0)
                .hint_text("runs/detect/train/weights/best.pt"),
        );
        if ui.button("选择…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("PyTorch 权重", &["pt", "onnx"])
                .pick_file()
            {
                state.test.weights = path.display().to_string();
            }
        }
        if ui.button("用训练得到的 best.pt").clicked() {
            if let Some(best) = &state.train.best {
                state.test.weights = best.display().to_string();
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("图片来源：");
        ui.add(
            egui::TextEdit::singleline(&mut state.test.source)
                .desired_width(280.0)
                .hint_text("图片文件或目录"),
        );
        if ui.button("选择图片…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("图片", &["jpg", "jpeg", "png", "bmp", "webp"])
                .pick_file()
            {
                state.test.source = path.display().to_string();
            }
        }
        if ui.button("选择目录…").clicked() {
            if let Some(dir) = rfd::FileDialog::new()
                .set_title("选择图片目录")
                .pick_folder()
            {
                state.test.source = dir.display().to_string();
            }
        }
    });

    ui.horizontal_wrapped(|ui| {
        ui.label("尺寸：");
        ui.add(
            egui::DragValue::new(&mut state.test.imgsz)
                .speed(32.0)
                .range(64..=2048),
        );
        ui.label("置信度：");
        ui.add(
            egui::DragValue::new(&mut state.test.conf)
                .speed(0.01)
                .range(0.0..=1.0),
        );
        ui.label("IoU：");
        ui.add(
            egui::DragValue::new(&mut state.test.iou)
                .speed(0.01)
                .range(0.0..=1.0),
        );
        ui.label("最大框数：");
        ui.add(
            egui::DragValue::new(&mut state.test.max_det)
                .speed(10.0)
                .range(1..=3000),
        );
        ui.label("设备：");
        ui.add(
            egui::TextEdit::singleline(&mut state.test.device)
                .desired_width(80.0)
                .hint_text("0 / cpu"),
        );
        ui.checkbox(&mut state.test.save, "保存标注图");
        ui.checkbox(&mut state.test.txt, "导出 txt");
    });

    ui.horizontal(|ui| {
        if ui
            .add_enabled(!running, egui::Button::new("开始推理"))
            .clicked()
        {
            start_infer(state);
        }
        if ui.add_enabled(running, egui::Button::new("取消")).clicked() {
            state.test.job.cancel();
            state.test.status = "正在终止推理…".to_string();
        }
        ui.separator();
        status_chip(ui, state.test.job.status);
        if let Some(progress) = &state.test.job.progress {
            ui.add(
                egui::ProgressBar::new(progress.ratio())
                    .desired_width(160.0)
                    .desired_height(14.0)
                    .text(format!(
                        "{}/{}",
                        fmt_num(progress.current),
                        fmt_num(progress.total)
                    ))
                    .animate(running),
            );
        }
        if !state.test.summary.is_empty() {
            ui.label(&state.test.summary);
        } else if !state.test.status.is_empty() {
            ui.label(&state.test.status);
        }
    });
}

fn start_infer(state: &mut AppState) {
    let Some(root) = bridge::locate_engine_root() else {
        state.test.status = "未找到 engine 目录，请设置 FORGE_ENGINE_ROOT".to_string();
        return;
    };
    let python = state.resolve_python(&root);
    if let Err(err) = state.test.start(&python, &root) {
        state.test.status = format!("{err:#}");
    }
}

fn status_chip(ui: &mut Ui, status: JobStatus) {
    match status {
        JobStatus::Idle => ui.weak("空闲"),
        JobStatus::Running => ui.colored_label(Color32::from_rgb(90, 170, 240), "运行中"),
        JobStatus::Finished => ui.colored_label(Color32::from_rgb(110, 190, 110), "已完成"),
        JobStatus::Failed => ui.colored_label(Color32::from_rgb(230, 120, 110), "失败"),
    };
}

fn result_list(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("推理结果").strong());
        ui.weak(format!("{} 张", state.test.results.len()));
    });
    ui.separator();

    if state.test.results.is_empty() {
        ui.weak("尚无推理结果。");
        return;
    }

    let mut chosen = None;
    egui::ScrollArea::vertical()
        .id_salt("test_result_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (index, result) in state.test.results.iter().enumerate() {
                let name = result.file_name();
                let count = result.detections.len();
                let ms = result
                    .inference_ms()
                    .map(|value| format!("{value:.1} ms"))
                    .unwrap_or_default();
                let text = format!("{name}  ·  {count} 框  ·  {ms}");
                let selected = state.test.selected == Some(index);
                let response = ui.selectable_label(selected, text);
                if response.clicked() {
                    chosen = Some(index);
                }
                if !result.detections.is_empty() {
                    let detail = result
                        .detections
                        .iter()
                        .map(|d| {
                            format!(
                                "{} {:.3}  xywhn=[{:.4}, {:.4}, {:.4}, {:.4}]",
                                d.name,
                                d.confidence,
                                d.xywhn[0],
                                d.xywhn[1],
                                d.xywhn[2],
                                d.xywhn[3]
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    response.on_hover_text(detail);
                }
            }
        });

    if let Some(index) = chosen {
        select_result(state, index);
    }
}

/// 选中一张结果并把预览图切过去（优先用引擎保存的标注图）。
fn select_result(state: &mut AppState, index: usize) {
    state.test.selected = Some(index);
    let Some(result) = state.test.results.get(index) else {
        state.test.preview.clear();
        return;
    };
    let path = result
        .annotated
        .clone()
        .unwrap_or_else(|| result.image.clone());
    state.test.preview.show(path);
}

fn preview(ui: &mut Ui, state: &mut AppState) {
    let Some(result) = state.test.selected_result().cloned() else {
        ui.centered_and_justified(|ui| ui.weak("推理完成后，从左侧选择一张图片查看结果。"));
        return;
    };

    // 检查过的图片只需解码一次，这里先确保纹理就绪。
    if state.test.preview.path().is_none() {
        let path = result
            .annotated
            .clone()
            .unwrap_or_else(|| result.image.clone());
        state.test.preview.show(path);
    }
    let ctx = ui.ctx().clone();
    let texture = state
        .test
        .preview
        .ensure(&ctx)
        .map(|image| (image.texture.id(), image.size()));
    let error = state.test.preview.error.clone();

    let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
    let canvas = response.rect;
    painter.rect_filled(canvas, 0.0, Color32::from_gray(24));

    let Some((texture, image_size)) = texture else {
        let message = error.unwrap_or_else(|| "正在加载图片…".to_string());
        painter.text(
            canvas.center(),
            Align2::CENTER_CENTER,
            message,
            FontId::proportional(15.0),
            Color32::LIGHT_GRAY,
        );
        return;
    };
    if image_size.x <= 0.0 || image_size.y <= 0.0 {
        return;
    }

    let scale = (canvas.width() / image_size.x).min(canvas.height() / image_size.y);
    let image_rect = Rect::from_center_size(canvas.center(), image_size * scale);
    painter.image(
        texture,
        image_rect,
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    );

    // 引擎已经画好框的标注图不用重复绘制。
    if result.annotated.is_none() {
        draw_detections(&painter, image_rect, &result);
    }

    let info = format!(
        "{}  ·  {}×{}  ·  {} 个目标",
        result.file_name(),
        result.width,
        result.height,
        result.detections.len()
    );
    painter.text(
        image_rect.left_top() + egui::vec2(8.0, 6.0),
        Align2::LEFT_TOP,
        info,
        FontId::proportional(13.0),
        Color32::from_gray(235),
    );
}

fn draw_detections(painter: &egui::Painter, image_rect: Rect, result: &Inference) {
    let scale_x = image_rect.width() / result.width.max(1) as f32;
    let scale_y = image_rect.height() / result.height.max(1) as f32;
    for detection in &result.detections {
        let rect = detection_rect(image_rect, detection, scale_x, scale_y);
        let color = crate::ui::class_color(detection.class_id);
        painter.rect_stroke(rect, 0.0, Stroke::new(1.8, color), StrokeKind::Outside);
        let label = format!("{} {:.2}", detection.name, detection.confidence);
        let galley = painter.layout_no_wrap(label, FontId::proportional(12.0), color);
        let anchor = rect.left_top() - egui::vec2(0.0, galley.size().y);
        painter.rect_filled(
            Rect::from_min_size(anchor, galley.size()),
            2.0,
            Color32::from_black_alpha(190),
        );
        painter.galley(anchor, galley, color);
    }
}

fn detection_rect(image_rect: Rect, detection: &Detection, scale_x: f32, scale_y: f32) -> Rect {
    let [x1, y1, x2, y2] = detection.xyxy;
    Rect::from_min_max(
        pos2(
            image_rect.left() + x1 * scale_x,
            image_rect.top() + y1 * scale_y,
        ),
        pos2(
            image_rect.left() + x2 * scale_x,
            image_rect.top() + y2 * scale_y,
        ),
    )
}

fn val_form(ui: &mut Ui, state: &mut AppState) {
    let running = state.test.val_job.is_running();

    ui.label(RichText::new("验证集评估").strong());
    ui.label("用 val 子命令统计 precision / recall / mAP。");

    ui.horizontal(|ui| {
        ui.label("数据集：");
        ui.add(
            egui::TextEdit::singleline(&mut state.test.dataset_root)
                .desired_width(200.0)
                .hint_text("datasets/hands_gesture"),
        );
        if ui.button("选择…").clicked() {
            if let Some(dir) = rfd::FileDialog::new()
                .set_title("选择数据集根目录")
                .pick_folder()
            {
                state.test.dataset_root = dir.display().to_string();
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("data.yaml：");
        ui.add(
            egui::TextEdit::singleline(&mut state.test.data_yaml)
                .desired_width(200.0)
                .hint_text("留空则自动生成"),
        );
        if ui.button("选择…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("YAML", &["yaml", "yml"])
                .pick_file()
            {
                state.test.data_yaml = path.display().to_string();
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("数据划分：");
        egui::ComboBox::from_id_salt("test_val_split")
            .width(110.0)
            .selected_text(state.test.split.clone())
            .show_ui(ui, |ui| {
                for split in VAL_SPLITS {
                    ui.selectable_value(&mut state.test.split, (*split).to_string(), *split);
                }
            });
        ui.label("尺寸：");
        ui.add(
            egui::DragValue::new(&mut state.test.imgsz)
                .speed(32.0)
                .range(64..=2048),
        );
    });

    ui.horizontal(|ui| {
        if ui
            .add_enabled(!running, egui::Button::new("开始评估"))
            .clicked()
        {
            if let Some(root) = bridge::locate_engine_root() {
                let python = state.resolve_python(&root);
                if let Err(err) = state.test.start_val(&python, &root) {
                    state.test.val_status = format!("{err:#}");
                }
            } else {
                state.test.val_status = "未找到 engine 目录，请设置 FORGE_ENGINE_ROOT".to_string();
            }
        }
        if ui.add_enabled(running, egui::Button::new("取消")).clicked() {
            state.test.val_job.cancel();
            state.test.val_status = "正在终止评估…".to_string();
        }
        ui.separator();
        status_chip(ui, state.test.val_job.status);
    });

    if running {
        if let Some(progress) = &state.test.val_job.progress {
            ui.add(
                egui::ProgressBar::new(progress.ratio())
                    .desired_height(12.0)
                    .animate(true),
            );
        }
    }
    if !state.test.val_status.is_empty() {
        ui.label(&state.test.val_status);
    }
}

fn val_metrics(ui: &mut Ui, state: &mut AppState) {
    let Some(metrics) = state.test.val_metrics.clone() else {
        ui.weak("尚未完成评估。");
        if !state.test.val_summary.is_empty() {
            ui.label(&state.test.val_summary);
        }
        return;
    };

    let rows: [(&str, &[&str]); 4] = [
        ("precision", &["precision", "metrics/precision(B)"]),
        ("recall", &["recall", "metrics/recall(B)"]),
        ("mAP50", &["mAP50", "metrics/mAP50(B)"]),
        ("mAP50-95", &["mAP50-95", "metrics/mAP50-95(B)"]),
    ];
    egui::Grid::new("test_val_grid")
        .num_columns(2)
        .striped(true)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            for (label, prefixes) in rows {
                let Some(value) = metrics.lookup(prefixes) else {
                    continue;
                };
                ui.label(label);
                ui.monospace(fmt_num(value));
                ui.end_row();
            }
        });

    if !metrics.per_class.is_empty() {
        ui.add_space(4.0);
        ui.label(RichText::new("各类别 mAP50-95").strong());
        egui::Grid::new("test_val_per_class")
            .num_columns(3)
            .striped(true)
            .spacing([12.0, 3.0])
            .show(ui, |ui| {
                for class in &metrics.per_class {
                    ui.weak(class.class_id.to_string());
                    ui.label(
                        RichText::new(&class.name).color(crate::ui::class_color(class.class_id)),
                    );
                    ui.monospace(fmt_num(class.map));
                    ui.end_row();
                }
            });
    }

    if !state.test.val_summary.is_empty() {
        ui.add_space(4.0);
        ui.label(&state.test.val_summary);
    }
    if let Some(err) = state.test.val_job.error.clone() {
        ui.colored_label(Color32::from_rgb(230, 120, 110), err);
    }
}

fn log_panel(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("运行日志").strong());
        ui.separator();
        let lines = state.test.job.logs.len() + state.test.val_job.logs.len();
        ui.weak(format!(
            "{}；{} · 共 {lines} 行",
            state.test.job.title(),
            state.test.val_job.kind()
        ));
        if ui.small_button("清空").clicked() {
            state.test.job.logs.clear();
            state.test.val_job.logs.clear();
        }
    });
    ui.separator();
    let default_color = ui.visuals().text_color();
    egui::ScrollArea::vertical()
        .id_salt("test_log_scroll")
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            if state.test.job.logs.is_empty() && state.test.val_job.logs.is_empty() {
                ui.weak("暂无日志。");
            }
            for line in state
                .test
                .job
                .logs
                .iter()
                .chain(state.test.val_job.logs.iter())
            {
                let color = match line.level {
                    bridge::LogLevel::Info => default_color,
                    bridge::LogLevel::Warn => Color32::from_rgb(240, 180, 60),
                    bridge::LogLevel::Error => Color32::from_rgb(230, 120, 110),
                };
                ui.monospace(RichText::new(&line.text).color(color));
            }
        });
}

/// 供外部打开文件所在目录。
#[allow(dead_code)]
pub fn open_dir(path: &Path) -> std::io::Result<()> {
    crate::ui::train::open_dir(path)
}
