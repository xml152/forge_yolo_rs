//! 各页面 UI。

use egui::Ui;

use crate::state::{AppState, Page};

pub mod annotate;
pub mod convert;
pub mod test;
pub mod train;

/// 按类别分配一个稳定的颜色：黄金分割让相邻色相尽量分散。
pub fn class_color(class_id: u32) -> egui::Color32 {
    let hue = (class_id as f32 * 0.618_034).fract();
    egui::Color32::from(egui::ecolor::Hsva::new(hue, 0.72, 0.95, 1.0))
}

pub fn top_bar(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.heading("Forge YOLO");
        ui.separator();
        for page in Page::ALL {
            if ui
                .selectable_label(state.page == page, page.title())
                .clicked()
            {
                state.page = page;
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut state.python)
                    .desired_width(300.0)
                    .hint_text("Python 解释器（留空自动查找）"),
            );
            ui.label("Python:");
        });
    });
}
