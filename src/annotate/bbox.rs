//! 标注框在“像素坐标”与“归一化坐标”之间的换算，以及拖拽命中检测。

use egui::{Pos2, Rect, Vec2};

use crate::yolo::label::YoloBox;

/// 拖拽手柄（框角）的命中半径，单位为像素。
pub const HANDLE_RADIUS: f32 = 7.0;

/// 新建框的最小像素尺寸，小于该值的框会被丢弃。
pub const MIN_BOX_SIZE: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corner {
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragMode {
    Move,
    Resize(Corner),
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub index: usize,
    pub mode: DragMode,
}

/// 归一化框 -> 图像显示区域内的像素矩形。
pub fn box_rect(b: &YoloBox, image_rect: Rect) -> Rect {
    let (x1, y1, x2, y2) = b.corners();
    Rect::from_min_max(
        Pos2::new(
            image_rect.left() + x1 * image_rect.width(),
            image_rect.top() + y1 * image_rect.height(),
        ),
        Pos2::new(
            image_rect.left() + x2 * image_rect.width(),
            image_rect.top() + y2 * image_rect.height(),
        ),
    )
}

/// 像素矩形 -> 归一化框。
pub fn rect_to_box(class_id: u32, rect: Rect, image_rect: Rect) -> YoloBox {
    if image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
        return YoloBox::new(class_id, 0.0, 0.0, 0.0, 0.0);
    }
    let x1 = ((rect.left() - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
    let y1 = ((rect.top() - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
    let x2 = ((rect.right() - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
    let y2 = ((rect.bottom() - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0);
    YoloBox::from_corners(class_id, x1, y1, x2, y2)
}

pub fn is_usable(rect: Rect) -> bool {
    rect.width() >= MIN_BOX_SIZE && rect.height() >= MIN_BOX_SIZE
}

/// 从后往前做命中检测：后绘制的框位于上层，优先命中。
pub fn hit_test(boxes: &[YoloBox], image_rect: Rect, pos: Pos2, handle: f32) -> Option<Hit> {
    for (index, b) in boxes.iter().enumerate().rev() {
        let r = box_rect(b, image_rect);
        let corners = [
            (Corner::NorthWest, r.left_top()),
            (Corner::NorthEast, r.right_top()),
            (Corner::SouthWest, r.left_bottom()),
            (Corner::SouthEast, r.right_bottom()),
        ];
        for (corner, p) in corners {
            if (pos - p).length() <= handle {
                return Some(Hit {
                    index,
                    mode: DragMode::Resize(corner),
                });
            }
        }
        if r.contains(pos) {
            return Some(Hit {
                index,
                mode: DragMode::Move,
            });
        }
    }
    None
}

/// 根据拖拽模式，把起始矩形按位移转换成新的矩形（限制在图像区域内部）。
pub fn apply_drag(start: Rect, mode: DragMode, delta: Vec2, image_rect: Rect) -> Rect {
    match mode {
        DragMode::Move => {
            let size = start.size();
            let mut min = start.min + delta;
            min.x = min.x.clamp(
                image_rect.left(),
                (image_rect.right() - size.x).max(image_rect.left()),
            );
            min.y = min.y.clamp(
                image_rect.top(),
                (image_rect.bottom() - size.y).max(image_rect.top()),
            );
            Rect::from_min_size(min, size)
        }
        DragMode::Resize(corner) => {
            let mut r = start;
            match corner {
                Corner::NorthWest => {
                    r.min.x += delta.x;
                    r.min.y += delta.y;
                }
                Corner::NorthEast => {
                    r.max.x += delta.x;
                    r.min.y += delta.y;
                }
                Corner::SouthWest => {
                    r.min.x += delta.x;
                    r.max.y += delta.y;
                }
                Corner::SouthEast => {
                    r.max.x += delta.x;
                    r.max.y += delta.y;
                }
            }
            // from_two_pos 会处理越过对边导致的坐标翻转。
            Rect::from_two_pos(r.min, r.max).intersect(image_rect)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_rect() -> Rect {
        Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(200.0, 100.0))
    }

    #[test]
    fn rect_roundtrip() {
        let b = YoloBox::new(3, 0.5, 0.5, 0.4, 0.6);
        let r = box_rect(&b, image_rect());
        assert!((r.center().x - 110.0).abs() < 1e-3);
        assert!((r.center().y - 70.0).abs() < 1e-3);

        let back = rect_to_box(3, r, image_rect());
        assert!((back.cx - 0.5).abs() < 1e-5);
        assert!((back.cy - 0.5).abs() < 1e-5);
        assert!((back.w - 0.4).abs() < 1e-5);
        assert!((back.h - 0.6).abs() < 1e-5);
    }

    #[test]
    fn move_is_clamped_inside_image() {
        let start = Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(50.0, 40.0));
        let moved = apply_drag(
            start,
            DragMode::Move,
            Vec2::new(-100.0, 500.0),
            image_rect(),
        );
        assert!(moved.left() >= image_rect().left() - 1e-3);
        assert!(moved.bottom() <= image_rect().bottom() + 1e-3);
        assert!((moved.width() - 50.0).abs() < 1e-3);
        assert!((moved.height() - 40.0).abs() < 1e-3);
    }

    #[test]
    fn resize_past_opposite_side_flips() {
        let start = Rect::from_min_size(Pos2::new(50.0, 50.0), Vec2::new(40.0, 40.0));
        let r = apply_drag(
            start,
            DragMode::Resize(Corner::NorthWest),
            Vec2::new(80.0, 80.0),
            Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(300.0, 300.0)),
        );
        assert!(r.width() > 0.0 && r.height() > 0.0);
    }

    #[test]
    fn hit_test_prefers_topmost_and_corners() {
        let boxes = vec![
            YoloBox::new(0, 0.5, 0.5, 0.5, 0.5),
            YoloBox::new(1, 0.5, 0.5, 0.5, 0.5),
        ];
        let rect = image_rect();
        let hit = hit_test(&boxes, rect, rect.center(), HANDLE_RADIUS).unwrap();
        assert_eq!(hit.index, 1);
        assert_eq!(hit.mode, DragMode::Move);

        // 两个框重合时，角点必然命中上层的那个；把下层框挪开再验证。
        let boxes = vec![
            YoloBox::new(0, 0.25, 0.25, 0.2, 0.2),
            YoloBox::new(1, 0.75, 0.75, 0.2, 0.2),
        ];
        let corner = box_rect(&boxes[0], rect).left_top();
        let hit = hit_test(&boxes, rect, corner, HANDLE_RADIUS).unwrap();
        assert_eq!(hit.index, 0);
        assert_eq!(hit.mode, DragMode::Resize(Corner::NorthWest));
    }
}
