//! 标注画布核心：当前数据集、当前图片、框编辑与拖拽状态。

pub mod bbox;
pub mod image;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use egui::{Pos2, Rect, Vec2};

use crate::yolo::dataset::{Dataset, DatasetStats};
use crate::yolo::label::YoloBox;

use self::bbox::{DragMode, HANDLE_RADIUS};
use self::image::LoadedImage;

#[derive(Default)]
pub struct Annotator {
    pub dataset: Option<Dataset>,
    pub stats: DatasetStats,
    pub index: usize,
    pub boxes: Vec<YoloBox>,
    pub selected: Option<usize>,
    /// 新建框时使用的类别。
    pub class_id: u32,
    pub dirty: bool,
    pub status: String,
    pub image_error: Option<String>,
    loaded: Option<LoadedImage>,
    drag: Option<DragState>,
    undo: Vec<Vec<YoloBox>>,
}

struct DragState {
    /// 已有框的索引；新建框时为 None。
    index: Option<usize>,
    mode: DragMode,
    start_pos: Pos2,
    start_rect: Rect,
    start_boxes: Vec<YoloBox>,
    image_rect: Rect,
    is_new: bool,
}

impl Annotator {
    pub fn open_dataset(&mut self, root: impl Into<PathBuf>) -> Result<()> {
        let dataset = Dataset::open(root)?;
        if dataset.is_empty() {
            anyhow::bail!("目录中没有找到图片：{}", dataset.root.display());
        }
        self.stats = dataset.stats();
        self.status = format!(
            "已加载 {} 张图片、{} 个类别",
            dataset.len(),
            dataset.classes.len()
        );
        self.dataset = Some(dataset);
        self.index = 0;
        self.load_current();
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        self.dataset.is_some()
    }

    pub fn total(&self) -> usize {
        self.dataset.as_ref().map_or(0, Dataset::len)
    }

    pub fn image_name(&self) -> String {
        self.dataset
            .as_ref()
            .and_then(|d| d.entries.get(self.index))
            .map(|e| e.name())
            .unwrap_or_default()
    }

    pub fn image_path(&self) -> Option<&Path> {
        self.dataset
            .as_ref()
            .and_then(|d| d.entries.get(self.index))
            .map(|e| e.image.as_path())
    }

    pub fn loaded(&self) -> Option<&LoadedImage> {
        self.loaded.as_ref()
    }

    /// 按需加载当前图片纹理（每帧调用即可）。
    pub fn ensure_texture(&mut self, ctx: &egui::Context) {
        if self.loaded.is_some() || self.image_error.is_some() {
            return;
        }
        let Some(path) = self.image_path().map(Path::to_path_buf) else {
            return;
        };
        match LoadedImage::load(ctx, &path) {
            Ok(img) => self.loaded = Some(img),
            Err(err) => self.image_error = Some(format!("{err:#}")),
        }
    }

    fn load_current(&mut self) {
        self.loaded = None;
        self.image_error = None;
        self.drag = None;
        self.undo.clear();
        self.selected = None;
        self.dirty = false;
        self.boxes = self
            .dataset
            .as_ref()
            .and_then(|d| d.entries.get(self.index))
            .map(|e| e.read_boxes())
            .unwrap_or_default();
    }

    /// 切换图片；若有未保存修改会先自动保存。
    pub fn goto(&mut self, index: usize) {
        let total = self.total();
        if total == 0 || index >= total || index == self.index {
            return;
        }
        if self.dirty {
            if let Err(err) = self.save() {
                self.status = format!("自动保存失败：{err:#}");
                return;
            }
        }
        self.index = index;
        self.load_current();
        self.status = format!("{}/{}  {}", self.index + 1, total, self.image_name());
    }

    pub fn next(&mut self) {
        self.goto(self.index + 1);
    }

    pub fn prev(&mut self) {
        self.goto(self.index.saturating_sub(1));
    }

    pub fn save(&mut self) -> Result<()> {
        let entry = self
            .dataset
            .as_ref()
            .context("尚未打开数据集")?
            .entries
            .get(self.index)
            .context("当前没有图片")?;
        entry
            .write_boxes(&self.boxes)
            .with_context(|| format!("写入标注失败：{}", entry.label.display()))?;
        self.dirty = false;
        self.status = format!("已保存 {}", entry.label.display());
        Ok(())
    }

    pub fn add_box(&mut self) {
        self.snapshot();
        self.boxes
            .push(YoloBox::new(self.class_id, 0.5, 0.5, 0.2, 0.2));
        self.selected = Some(self.boxes.len() - 1);
    }

    pub fn delete_selected(&mut self) {
        let Some(index) = self.selected else { return };
        if index >= self.boxes.len() {
            return;
        }
        self.snapshot();
        self.boxes.remove(index);
        self.selected = if self.boxes.is_empty() {
            None
        } else {
            Some(index.min(self.boxes.len() - 1))
        };
    }

    /// 设置“新建框”使用的类别；如果当前有选中框，也一并修改它的类别。
    pub fn choose_class(&mut self, class_id: u32) {
        self.class_id = class_id;
        if let Some(index) = self.selected {
            if self
                .boxes
                .get(index)
                .is_some_and(|b| b.class_id != class_id)
            {
                self.snapshot();
                self.boxes[index].class_id = class_id;
            }
        }
    }

    pub fn undo(&mut self) {
        if let Some(previous) = self.undo.pop() {
            self.boxes = previous;
            self.selected = None;
            self.dirty = true;
            self.status = "已撤销上一步".to_string();
        }
    }

    pub fn select_at(&mut self, pos: Pos2, image_rect: Rect) {
        self.selected =
            bbox::hit_test(&self.boxes, image_rect, pos, HANDLE_RADIUS).map(|h| h.index);
    }

    pub fn begin_drag(&mut self, pos: Pos2, image_rect: Rect) {
        if let Some(hit) = bbox::hit_test(&self.boxes, image_rect, pos, HANDLE_RADIUS) {
            self.selected = Some(hit.index);
            self.drag = Some(DragState {
                index: Some(hit.index),
                mode: hit.mode,
                start_pos: pos,
                start_rect: bbox::box_rect(&self.boxes[hit.index], image_rect),
                start_boxes: self.boxes.clone(),
                image_rect,
                is_new: false,
            });
        } else {
            self.selected = None;
            self.drag = Some(DragState {
                index: None,
                mode: DragMode::Move,
                start_pos: pos,
                start_rect: Rect::from_min_size(pos, Vec2::ZERO),
                start_boxes: self.boxes.clone(),
                image_rect,
                is_new: true,
            });
        }
    }

    pub fn update_drag(&mut self, pos: Pos2) {
        let Some(drag) = &self.drag else { return };
        let (mode, start_pos, start_rect, image_rect, is_new, index, start_len) = (
            drag.mode,
            drag.start_pos,
            drag.start_rect,
            drag.image_rect,
            drag.is_new,
            drag.index,
            drag.start_boxes.len(),
        );

        if is_new {
            let rect = Rect::from_two_pos(start_pos, pos).intersect(image_rect);
            let new_box = bbox::rect_to_box(self.class_id, rect, image_rect);
            if self.boxes.len() == start_len {
                self.boxes.push(new_box);
            } else if let Some(slot) = self.boxes.get_mut(start_len) {
                *slot = new_box;
            }
            self.selected = Some(start_len);
        } else if let Some(index) = index {
            let rect = bbox::apply_drag(start_rect, mode, pos - start_pos, image_rect);
            let class_id = self.boxes.get(index).map_or(self.class_id, |b| b.class_id);
            let updated = bbox::rect_to_box(class_id, rect, image_rect);
            if let Some(slot) = self.boxes.get_mut(index) {
                *slot = updated;
            }
        }
    }

    pub fn end_drag(&mut self) {
        let Some(drag) = self.drag.take() else { return };

        if drag.is_new {
            let new_index = drag.start_boxes.len();
            let keep = self
                .boxes
                .get(new_index)
                .is_some_and(|b| bbox::is_usable(bbox::box_rect(b, drag.image_rect)));
            if !keep {
                self.boxes.truncate(drag.start_boxes.len());
                if self.selected == Some(new_index) {
                    self.selected = None;
                }
            }
        }

        if self.boxes != drag.start_boxes {
            self.push_undo(drag.start_boxes);
            self.dirty = true;
        }
    }

    fn snapshot(&mut self) {
        let previous = self.boxes.clone();
        self.push_undo(previous);
        self.dirty = true;
    }

    fn push_undo(&mut self, previous: Vec<YoloBox>) {
        self.undo.push(previous);
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
    }
}
