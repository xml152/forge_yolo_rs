//! 数据标注页面：画布 + 类别面板 + 快捷键。

use egui::{Align2, Color32, FontId, Key, Modifiers, Rect, Sense, Stroke, StrokeKind, Ui, pos2};

use crate::annotate::bbox;
use crate::state::AppState;

/// 数字键直接切换类别（Num0 对应第 10 个类别）。
const NUM_KEYS: [Key; 10] = [
    Key::Num0,
    Key::Num1,
    Key::Num2,
    Key::Num3,
    Key::Num4,
    Key::Num5,
    Key::Num6,
    Key::Num7,
    Key::Num8,
    Key::Num9,
];

pub fn show(ui: &mut Ui, state: &mut AppState) {
    handle_keys(ui, state);

    egui::Panel::top("annotate_toolbar").show(ui, |ui| {
        ui.add_space(2.0);
        toolbar(ui, state);
        ui.add_space(2.0);
    });
    egui::Panel::left("annotate_classes")
        .resizable(true)
        .default_size(190.0)
        .show(ui, |ui| class_panel(ui, state));
    egui::CentralPanel::default().show(ui, |ui| canvas(ui, state));
}

fn toolbar(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label("数据集：");
        ui.add(
            egui::TextEdit::singleline(&mut state.dataset_root)
                .desired_width(300.0)
                .hint_text("datasets/hands_gesture"),
        );
        if ui.button("打开").clicked() {
            let root = state.dataset_root.clone();
            open_dataset(state, &root);
        }
        if ui.button("浏览…").clicked() {
            if let Some(dir) = rfd::FileDialog::new()
                .set_title("选择数据集根目录")
                .pick_folder()
            {
                state.dataset_root = dir.display().to_string();
                open_dataset(state, &dir.display().to_string());
            }
        }
    });

    ui.horizontal(|ui| {
        let total = state.annotator.total();
        ui.add_enabled_ui(total > 0, |ui| {
            if ui.button("◀ 上一张 (A)").clicked() {
                state.annotator.prev();
            }
            if ui.button("下一张 (D) ▶").clicked() {
                state.annotator.next();
            }
        });
        ui.separator();
        if ui.button("保存 (Ctrl+S)").clicked() {
            save(state);
        }
        if ui.button("撤销 (Ctrl+Z)").clicked() {
            state.annotator.undo();
        }
        ui.separator();
        if ui.button("添加框 (N)").clicked() {
            state.annotator.add_box();
        }
        if ui.button("删除框 (Del)").clicked() {
            state.annotator.delete_selected();
        }
        ui.separator();
        let class_id = state.annotator.class_id;
        let label = state
            .annotator
            .dataset
            .as_ref()
            .map(|d| d.class_name(class_id))
            .unwrap_or_else(|| format!("class_{class_id}"));
        ui.label("当前类别：");
        egui::ComboBox::from_id_salt("annotate_class")
            .width(180.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                let classes = state
                    .annotator
                    .dataset
                    .as_ref()
                    .map(|d| d.classes.clone())
                    .unwrap_or_default();
                for (index, name) in classes.iter().enumerate() {
                    let id = index as u32;
                    ui.selectable_value(
                        &mut state.annotator.class_id,
                        id,
                        egui::RichText::new(format!("{index:>2}. {name}"))
                            .color(crate::ui::class_color(id)),
                    );
                }
            });
    });

    ui.horizontal(|ui| {
        if state.annotator.is_open() {
            let index = state.annotator.index + 1;
            let total = state.annotator.total();
            ui.label(format!("{index} / {total}"));
            ui.separator();
            let name = ui.monospace(state.annotator.image_name());
            if let Some(path) = state.annotator.loaded().map(|image| image.path.clone()) {
                name.on_hover_text(path.display().to_string());
            }
            ui.separator();
            ui.label(format!("{} 个框", state.annotator.boxes.len()));
            if state.annotator.dirty {
                ui.colored_label(Color32::from_rgb(240, 180, 60), "未保存");
            }
        } else {
            ui.weak("尚未打开数据集");
        }
        ui.separator();
        ui.weak(&state.annotator.status);
    });
}

fn open_dataset(state: &mut AppState, root: &str) {
    let root = root.trim().to_string();
    if root.is_empty() {
        state.annotator.status = "请先填写数据集目录".to_string();
        return;
    }
    match state.annotator.open_dataset(&root) {
        Ok(()) => {}
        Err(err) => state.annotator.status = format!("打开数据集失败：{err:#}"),
    }
}

fn save(state: &mut AppState) {
    if let Err(err) = state.annotator.save() {
        state.annotator.status = format!("保存失败：{err:#}");
    }
}

fn class_panel(ui: &mut Ui, state: &mut AppState) {
    let (classes, counts, stats, warnings) = match state.annotator.dataset.as_ref() {
        Some(dataset) => (
            dataset.classes.clone(),
            state.annotator.stats.per_class.clone(),
            state.annotator.stats.clone(),
            dataset.warnings.clone(),
        ),
        None => {
            ui.add_space(6.0);
            ui.weak("未打开数据集");
            return;
        }
    };

    ui.add_space(4.0);
    ui.label(egui::RichText::new("类别").strong());
    ui.separator();

    egui::ScrollArea::vertical()
        .id_salt("annotate_class_list")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (index, name) in classes.iter().enumerate() {
                let id = index as u32;
                let count = counts.get(index).copied().unwrap_or(0);
                let selected = state.annotator.class_id == id;
                let text = egui::RichText::new(format!("{index:>2}. {name}  ({count})"))
                    .color(crate::ui::class_color(id));
                if ui.selectable_label(selected, text).clicked() {
                    state.annotator.choose_class(id);
                }
            }
        });

    ui.separator();
    ui.label(format!("图片：{}", stats.images));
    ui.label(format!("已标注：{}", stats.labeled));
    ui.label(format!("总框数：{}", stats.boxes));
    if stats.labeled < stats.images {
        ui.colored_label(
            Color32::from_rgb(240, 180, 60),
            format!("未标注：{}", stats.images - stats.labeled),
        );
    }
    if !warnings.is_empty() {
        ui.separator();
        for warning in &warnings {
            ui.colored_label(Color32::from_rgb(240, 180, 60), warning);
        }
    }
    if let Some(dataset) = state.annotator.dataset.as_ref() {
        ui.separator();
        ui.weak("标签目录")
            .on_hover_text(dataset.labels_dir.display().to_string());
        ui.small(dataset.labels_dir.display().to_string());
    }
}

fn canvas(ui: &mut Ui, state: &mut AppState) {
    if state.annotator.dataset.is_none() {
        ui.centered_and_justified(|ui| ui.weak("请选择数据集目录后点击“打开”。"));
        return;
    }

    let ctx = ui.ctx().clone();
    state.annotator.ensure_texture(&ctx);

    let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
    let canvas_rect = response.rect;
    painter.rect_filled(canvas_rect, 0.0, Color32::from_gray(26));

    let loaded = state
        .annotator
        .loaded()
        .map(|image| (image.texture.id(), image.size()));

    let Some((texture, image_size)) = loaded else {
        let message = state
            .annotator
            .image_error
            .clone()
            .unwrap_or_else(|| "正在加载图片…".to_string());
        painter.text(
            canvas_rect.center(),
            Align2::CENTER_CENTER,
            message,
            FontId::proportional(15.0),
            Color32::from_gray(200),
        );
        return;
    };

    if image_size.x <= 0.0 || image_size.y <= 0.0 {
        return;
    }
    let scale = (canvas_rect.width() / image_size.x).min(canvas_rect.height() / image_size.y);
    let image_rect = Rect::from_center_size(canvas_rect.center(), image_size * scale);

    // 先处理输入，本帧立刻就能看到拖拽结果。
    let annotator = &mut state.annotator;
    let pointer = response.interact_pointer_pos();
    if let Some(pos) = pointer {
        if response.drag_started() && image_rect.contains(pos) {
            annotator.begin_drag(pos, image_rect);
        } else if response.dragged() {
            annotator.update_drag(pos);
        }
    }
    if response.drag_stopped() {
        annotator.end_drag();
    }
    if response.clicked() {
        if let Some(pos) = pointer {
            annotator.select_at(pos, image_rect);
        }
    }
    if response.hovered() {
        ctx.set_cursor_icon(if response.dragged() {
            egui::CursorIcon::Grabbing
        } else {
            egui::CursorIcon::Crosshair
        });
    }

    painter.image(
        texture,
        image_rect,
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    );

    let names = annotator
        .dataset
        .as_ref()
        .map(|d| d.classes.clone())
        .unwrap_or_default();

    for (index, b) in annotator.boxes.iter().enumerate() {
        let rect = bbox::box_rect(b, image_rect);
        let selected = annotator.selected == Some(index);
        let color = crate::ui::class_color(b.class_id);
        let stroke = if selected {
            Stroke::new(2.5, Color32::WHITE)
        } else {
            Stroke::new(1.6, color)
        };
        painter.rect_stroke(rect, 0.0, stroke, StrokeKind::Outside);

        if selected {
            for corner in [
                rect.left_top(),
                rect.right_top(),
                rect.left_bottom(),
                rect.right_bottom(),
            ] {
                painter.circle_filled(corner, bbox::HANDLE_RADIUS * 0.7, Color32::WHITE);
            }
        }

        if rect.width() > 18.0 {
            let name = names
                .get(b.class_id as usize)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", b.class_id));
            let galley = painter.layout_no_wrap(name, FontId::proportional(12.0), Color32::BLACK);
            let anchor = rect.left_top() - egui::vec2(0.0, galley.size().y);
            painter.rect_filled(
                Rect::from_min_size(anchor, galley.size()),
                2.0,
                color.gamma_multiply(0.9),
            );
            painter.galley(anchor, galley, Color32::BLACK);
        }
    }

    let info = format!(
        "{}/{}  ·  {}  ·  {} 个框",
        annotator.index + 1,
        annotator.total(),
        annotator.image_name(),
        annotator.boxes.len()
    );
    painter.text(
        canvas_rect.left_top() + egui::vec2(8.0, 8.0),
        Align2::LEFT_TOP,
        info,
        FontId::proportional(13.0),
        Color32::from_gray(220),
    );

    if !annotator.status.is_empty() {
        painter.text(
            canvas_rect.left_bottom() + egui::vec2(8.0, -8.0),
            Align2::LEFT_BOTTOM,
            &annotator.status,
            FontId::proportional(12.0),
            Color32::from_gray(180),
        );
    }
}

fn handle_keys(ui: &Ui, state: &mut AppState) {
    let ctx = ui.ctx().clone();
    // 焦点在输入框里时不抢按键。
    if ctx.egui_wants_keyboard_input() {
        return;
    }

    let mut save_pressed = false;
    let mut undo_pressed = false;
    let mut next_pressed = false;
    let mut prev_pressed = false;
    let mut delete_pressed = false;
    let mut add_pressed = false;
    let mut class_pressed = None;

    ctx.input_mut(|input| {
        save_pressed = input.consume_key(Modifiers::COMMAND, Key::S);
        undo_pressed = input.consume_key(Modifiers::COMMAND, Key::Z);
        next_pressed = input.consume_key(Modifiers::NONE, Key::D)
            || input.consume_key(Modifiers::NONE, Key::ArrowRight);
        prev_pressed = input.consume_key(Modifiers::NONE, Key::A)
            || input.consume_key(Modifiers::NONE, Key::ArrowLeft);
        delete_pressed = input.consume_key(Modifiers::NONE, Key::Delete)
            || input.consume_key(Modifiers::NONE, Key::Backspace);
        add_pressed = input.consume_key(Modifiers::NONE, Key::N);
        for (index, key) in NUM_KEYS.iter().enumerate() {
            if input.consume_key(Modifiers::NONE, *key) {
                // Num1 → 第 0 类，…，Num9 → 第 8 类，Num0 → 第 9 类。
                class_pressed = Some(((index + 9) % 10) as u32);
                break;
            }
        }
    });

    if !state.annotator.is_open() {
        return;
    }
    if save_pressed {
        save(state);
    }
    if undo_pressed {
        state.annotator.undo();
    }
    if delete_pressed {
        state.annotator.delete_selected();
    }
    if add_pressed {
        state.annotator.add_box();
    }
    if let Some(class_id) = class_pressed {
        state.annotator.choose_class(class_id);
    }
    if prev_pressed {
        state.annotator.prev();
    }
    if next_pressed {
        state.annotator.next();
    }
}
