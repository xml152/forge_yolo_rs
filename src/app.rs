//! 顶层 App：顶部工具栏 + 左侧导航 + 中央页面。

use std::time::Duration;

use eframe::Frame;
use egui::{CentralPanel, Panel, Ui};

use crate::state::{AppState, Page, Settings};
use crate::ui;

/// 界面配置在 eframe 存储里的键名。
const SETTINGS_KEY: &str = "forge_yolo_settings";

pub struct ForgeApp {
    state: AppState,
}

impl ForgeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_cjk_font(&cc.egui_ctx);
        let mut state = AppState::new();
        if let Some(settings) = cc
            .storage
            .and_then(|storage| eframe::get_value::<Settings>(storage, SETTINGS_KEY))
        {
            settings.apply(&mut state);
        }
        Self { state }
    }
}

impl eframe::App for ForgeApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut Frame) {
        Panel::top("forge_top_bar").show(ui, |ui| {
            ui::top_bar(ui, &mut self.state);
        });

        Panel::left("forge_nav")
            .resizable(false)
            .default_size(132.0)
            .show(ui, |ui| {
                ui.add_space(8.0);
                for page in Page::ALL {
                    let selected = self.state.page == page;
                    let button = egui::Button::new(page.title()).selected(selected);
                    if ui.add_sized([ui.available_width(), 34.0], button).clicked() {
                        self.state.page = page;
                    }
                }
            });

        let page = self.state.page;
        CentralPanel::default().show(ui, |ui| match page {
            Page::Annotate => ui::annotate::show(ui, &mut self.state),
            Page::Train => ui::train::show(ui, &mut self.state),
            Page::Convert => ui::convert::show(ui, &mut self.state),
            Page::Test => ui::test::show(ui, &mut self.state),
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, SETTINGS_KEY, &Settings::capture(&self.state));
    }

    /// 训练/推理可能跑很久，缩短自动保存间隔以免配置丢失。
    fn auto_save_interval(&self) -> Duration {
        Duration::from_secs(10)
    }
}

/// egui 自带字体不含中文字形，这里从系统中挑一个可用的 CJK 字体。
fn install_cjk_font(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/PingFangSC/pingfangsc-regular.otf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ];

    for path in CANDIDATES {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "cjk".to_string(),
            std::sync::Arc::new(egui::FontData::from_owned(bytes)),
        );
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "cjk".to_string());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .push("cjk".to_string());
        ctx.set_fonts(fonts);
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 无头渲染四个页面，任何布局/索引 panic 都会在这里暴露。
    #[test]
    fn every_page_renders_headless() {
        let ctx = egui::Context::default();
        let mut state = AppState::new();

        for page in Page::ALL {
            state.page = page;
            for _ in 0..2 {
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280.0, 800.0),
                    )),
                    ..Default::default()
                };
                let output = ctx.run_ui(input, |ui| {
                    Panel::top("forge_top_bar").show(ui, |ui| {
                        ui::top_bar(ui, &mut state);
                    });
                    Panel::left("forge_nav")
                        .resizable(false)
                        .default_size(132.0)
                        .show(ui, |ui| {
                            for item in Page::ALL {
                                ui.add_sized(
                                    [ui.available_width(), 34.0],
                                    egui::Button::new(item.title()).selected(item == page),
                                );
                            }
                        });
                    CentralPanel::default().show(ui, |ui| match page {
                        Page::Annotate => ui::annotate::show(ui, &mut state),
                        Page::Train => ui::train::show(ui, &mut state),
                        Page::Convert => ui::convert::show(ui, &mut state),
                        Page::Test => ui::test::show(ui, &mut state),
                    });
                });
                output.drop_without_applying_deltas();
            }
        }
    }
}
