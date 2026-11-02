//! YOLO 检测标注格式：
//! 每个 txt 文件与图片同名，每行 `class_id cx cy w h`，坐标均为 0~1 的归一化值。

use std::path::Path;

/// 一个归一化的检测框。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YoloBox {
    pub class_id: u32,
    /// 中心点 x（0~1）
    pub cx: f32,
    /// 中心点 y（0~1）
    pub cy: f32,
    /// 宽（0~1）
    pub w: f32,
    /// 高（0~1）
    pub h: f32,
}

impl YoloBox {
    pub fn new(class_id: u32, cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            class_id,
            cx,
            cy,
            w,
            h,
        }
    }

    /// 解析一行标注；分割/关键点等多余字段直接忽略。非法行返回 `None`。
    pub fn parse_line(line: &str) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let mut it = line.split_whitespace();
        let class_id: u32 = it.next()?.parse().ok()?;
        let cx: f32 = it.next()?.parse().ok()?;
        let cy: f32 = it.next()?.parse().ok()?;
        let w: f32 = it.next()?.parse().ok()?;
        let h: f32 = it.next()?.parse().ok()?;
        if ![cx, cy, w, h].iter().all(|v| v.is_finite()) {
            return None;
        }
        Some(Self {
            class_id,
            cx: cx.clamp(0.0, 1.0),
            cy: cy.clamp(0.0, 1.0),
            w: w.clamp(0.0, 1.0),
            h: h.clamp(0.0, 1.0),
        })
    }

    pub fn to_line(&self) -> String {
        format!(
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id, self.cx, self.cy, self.w, self.h
        )
    }

    /// 左上角与右下角（归一化）。
    pub fn corners(&self) -> (f32, f32, f32, f32) {
        (
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        )
    }

    /// 由归一化的左上/右下角构造，并夹紧到图像范围内。
    pub fn from_corners(class_id: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let (x1, x2) = (x1.min(x2).clamp(0.0, 1.0), x1.max(x2).clamp(0.0, 1.0));
        let (y1, y2) = (y1.min(y2).clamp(0.0, 1.0), y1.max(y2).clamp(0.0, 1.0));
        Self {
            class_id,
            cx: (x1 + x2) / 2.0,
            cy: (y1 + y2) / 2.0,
            w: x2 - x1,
            h: y2 - y1,
        }
    }
}

pub fn parse(text: &str) -> Vec<YoloBox> {
    text.lines().filter_map(YoloBox::parse_line).collect()
}

pub fn to_text(boxes: &[YoloBox]) -> String {
    let mut out = String::new();
    for b in boxes {
        out.push_str(&b.to_line());
        out.push('\n');
    }
    out
}

pub fn read_label_file(path: &Path) -> Vec<YoloBox> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse(&text),
        Err(_) => Vec::new(),
    }
}

pub fn write_label_file(path: &Path, boxes: &[YoloBox]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, to_text(boxes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let boxes = parse("0 0.5 0.5 0.2 0.4\n1 0.1 0.2 0.3 0.3\n");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].class_id, 0);
        assert!((boxes[0].cx - 0.5).abs() < 1e-6);
        assert_eq!(boxes[1].class_id, 1);
    }

    #[test]
    fn parse_skips_invalid_lines() {
        let boxes = parse("bad line\n0 0.5 0.5 0.2 0.4\n\n# comment\n1 0.1 0.2\n");
        assert_eq!(boxes.len(), 1);
    }

    #[test]
    fn roundtrip() {
        let b = YoloBox::new(3, 0.25, 0.75, 0.5, 0.25);
        let text = to_text(&[b]);
        let parsed = parse(&text);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0], b);
    }

    #[test]
    fn corners_and_clamp() {
        let b = YoloBox::from_corners(0, 0.8, 0.9, 0.2, 0.3);
        assert!((b.w - 0.6).abs() < 1e-6);
        assert!((b.h - 0.6).abs() < 1e-6);
        assert!((b.cx - 0.5).abs() < 1e-6);
        let (x1, y1, x2, y2) = b.corners();
        assert!((x1 - 0.2).abs() < 1e-6 && (y2 - 0.9).abs() < 1e-6);
        let _ = (x2, y1);
    }

    #[test]
    fn clamps_out_of_range() {
        let b = YoloBox::parse_line("0 1.5 0.5 0.2 0.4").unwrap();
        assert!((b.cx - 1.0).abs() < 1e-6);
    }
}
