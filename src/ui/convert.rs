//! 数据集格式转换页面。

use egui::Ui;

use crate::state::AppState;
use crate::yolo::convert::TargetFormat;

pub fn show(ui: &mut Ui, state: &mut AppState) {
    ui.heading("数据集格式转换");
    ui.label("把 YOLO txt 标注转换成其他训练框架需要的目录结构。");
    ui.add_space(8.0);

    let convert = &mut state.convert;

    ui.horizontal(|ui| {
        ui.label("源数据集：");
        ui.add(
            egui::TextEdit::singleline(&mut convert.dataset_root)
                .desired_width(360.0)
                .hint_text("datasets/hands_gesture"),
        );
        if ui.button("浏览…").clicked() {
            if let Some(dir) = rfd::FileDialog::new()
                .set_title("选择数据集根目录")
                .pick_folder()
            {
                convert.dataset_root = dir.display().to_string();
            }
        }
        if ui.button("用标注页的数据集").clicked() {
            if let Some(dataset) = &state.annotator.dataset {
                convert.dataset_root = dataset.root.display().to_string();
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("输出目录：");
        ui.add(
            egui::TextEdit::singleline(&mut convert.out_dir)
                .desired_width(360.0)
                .hint_text("runs/convert"),
        );
        ui.weak(format!(
            "→ {}",
            std::path::Path::new(&convert.out_dir)
                .join(convert.format.folder_name())
                .display()
        ));
        if ui.button("浏览…").clicked() {
            if let Some(dir) = rfd::FileDialog::new()
                .set_title("选择输出目录")
                .pick_folder()
            {
                convert.out_dir = dir.display().to_string();
            }
        }
    });

    ui.horizontal(|ui| {
        ui.label("目标格式：");
        egui::ComboBox::from_id_salt("convert_format")
            .width(260.0)
            .selected_text(convert.format.label())
            .show_ui(ui, |ui| {
                for format in TargetFormat::ALL {
                    ui.selectable_value(&mut convert.format, format, format.label());
                }
            });
    });

    ui.horizontal(|ui| {
        ui.label("验证集比例：");
        ui.add(
            egui::DragValue::new(&mut convert.val_ratio)
                .speed(0.01)
                .range(0.0..=0.9),
        );
        ui.checkbox(&mut convert.copy_images, "复制图片（否则只写标注与清单）");
    });

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let label = if convert.report.is_empty() {
            "开始转换".to_string()
        } else {
            "重新转换".to_string()
        };
        if ui.button(label).clicked() {
            match convert.run() {
                Ok(text) => convert.report = text,
                Err(err) => convert.report = format!("转换失败：{err:#}"),
            }
        }
        if ui.button("打开输出目录").clicked() {
            let path = std::path::Path::new(&convert.out_dir);
            if let Err(err) = opener(path) {
                convert.report = format!("无法打开目录：{err}");
            }
        }
    });

    ui.separator();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if convert.report.is_empty() {
                ui.weak("尚未执行转换。");
            } else {
                ui.monospace(&convert.report);
            }
        });
}

fn opener(path: &std::path::Path) -> std::io::Result<()> {
    if !path.exists() {
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
