//! 图片加载与纹理管理。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub struct LoadedImage {
    pub path: PathBuf,
    pub texture: egui::TextureHandle,
    pub width: usize,
    pub height: usize,
}

impl LoadedImage {
    pub fn load(ctx: &egui::Context, path: &Path) -> Result<Self> {
        let decoded = image::ImageReader::open(path)
            .with_context(|| format!("无法打开图片 {}", path.display()))?
            .with_guessed_format()
            .with_context(|| format!("无法识别图片格式 {}", path.display()))?
            .decode()
            .with_context(|| format!("无法解码图片 {}", path.display()))?;
        let rgba = decoded.to_rgba8();
        let (width, height) = rgba.dimensions();
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [width as usize, height as usize],
            rgba.as_raw(),
        );
        let texture = ctx.load_texture(path.to_string_lossy(), color, egui::TextureOptions::LINEAR);
        Ok(Self {
            path: path.to_path_buf(),
            texture,
            width: width as usize,
            height: height as usize,
        })
    }

    pub fn size(&self) -> egui::Vec2 {
        egui::vec2(self.width as f32, self.height as f32)
    }
}
