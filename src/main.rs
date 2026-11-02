mod annotate;
mod app;
mod bridge;
mod state;
mod ui;
mod yolo;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Forge YOLO")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Forge YOLO",
        options,
        Box::new(|cc| Ok(Box::new(app::ForgeApp::new(cc)))),
    )
}
