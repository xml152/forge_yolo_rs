//! 数据集扫描与类别管理。
//!
//! 支持两种常见目录布局：
//! ```text
//! dataset/
//!   images/xxx.jpg          (或图片直接放在根目录)
//!   labels/xxx.txt
//!   classes.txt | data.yaml
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::label::{self, YoloBox};

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            IMAGE_EXTENSIONS.contains(&e.as_str())
        })
        .unwrap_or(false)
}

#[derive(Clone, Debug)]
pub struct ImageEntry {
    pub image: PathBuf,
    pub label: PathBuf,
}

impl ImageEntry {
    pub fn name(&self) -> String {
        self.image
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    pub fn has_label(&self) -> bool {
        self.label.is_file()
    }

    pub fn read_boxes(&self) -> Vec<YoloBox> {
        label::read_label_file(&self.label)
    }

    pub fn write_boxes(&self, boxes: &[YoloBox]) -> std::io::Result<()> {
        label::write_label_file(&self.label, boxes)
    }
}

#[derive(Clone, Debug, Default)]
pub struct DatasetStats {
    pub images: usize,
    pub labeled: usize,
    pub boxes: usize,
    pub per_class: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub root: PathBuf,
    pub images_dir: PathBuf,
    pub labels_dir: PathBuf,
    pub classes: Vec<String>,
    pub entries: Vec<ImageEntry>,
    pub warnings: Vec<String>,
}

impl Dataset {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        if !root.is_dir() {
            anyhow::bail!("目录不存在：{}", root.display());
        }

        let images_dir = if root.join("images").is_dir() {
            root.join("images")
        } else {
            root.clone()
        };
        let labels_dir = if root.join("labels").is_dir() {
            root.join("labels")
        } else {
            match images_dir.file_name().and_then(|n| n.to_str()) {
                Some("images") => images_dir
                    .parent()
                    .map(|p| p.join("labels"))
                    .unwrap_or_else(|| root.join("labels")),
                _ => images_dir.clone(),
            }
        };

        let mut images = Vec::new();
        collect_images(&images_dir, &mut images)?;
        images.sort();

        let mut entries = Vec::with_capacity(images.len());
        for image in images {
            let rel = image
                .strip_prefix(&images_dir)
                .unwrap_or(&image)
                .to_path_buf();
            let label = labels_dir.join(rel).with_extension("txt");
            entries.push(ImageEntry { image, label });
        }

        let mut warnings = Vec::new();
        let mut classes = load_classes(&root, &images_dir);

        // 若类别名缺失或少于标注中的最大类别 id，自动补齐。
        let mut max_class: Option<u32> = None;
        for entry in &entries {
            for b in entry.read_boxes() {
                max_class = Some(max_class.map_or(b.class_id, |m| m.max(b.class_id)));
            }
        }
        if let Some(max_id) = max_class {
            let need = max_id as usize + 1;
            if classes.len() < need {
                warnings.push(format!(
                    "类别名数量（{}）少于标注中的最大类别 id + 1（{}），已用 class_N 补齐",
                    classes.len(),
                    need
                ));
                while classes.len() < need {
                    classes.push(format!("class_{}", classes.len()));
                }
            }
        }
        if classes.is_empty() {
            warnings.push("未找到类别名，已默认创建一个类别 class_0".to_string());
            classes.push("class_0".to_string());
        }

        Ok(Self {
            root,
            images_dir,
            labels_dir,
            classes,
            entries,
            warnings,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn class_name(&self, id: u32) -> String {
        self.classes
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| format!("class_{id}"))
    }

    pub fn stats(&self) -> DatasetStats {
        let mut stats = DatasetStats {
            images: self.entries.len(),
            per_class: vec![0; self.classes.len().max(1)],
            ..Default::default()
        };
        for entry in &self.entries {
            let boxes = entry.read_boxes();
            if entry.has_label() {
                stats.labeled += 1;
            }
            stats.boxes += boxes.len();
            for b in boxes {
                let idx = b.class_id as usize;
                if idx >= stats.per_class.len() {
                    stats.per_class.resize(idx + 1, 0);
                }
                stats.per_class[idx] += 1;
            }
        }
        stats
    }

    /// 生成 ultralytics 训练用的 data.yaml 内容。
    ///
    /// `val_split` 为 true 时验证集复用训练集（数据量太少、没有单独验证集的场景）。
    pub fn data_yaml(&self, val_split: bool) -> String {
        let rel = |path: &Path| -> String {
            match path.strip_prefix(&self.root) {
                Ok(rest) if !rest.as_os_str().is_empty() => {
                    rest.to_string_lossy().replace('\\', "/")
                }
                _ => ".".to_string(),
            }
        };

        // 兼容 `images/train` + `images/val` 的划分式布局。
        let train_dir = if self.images_dir.join("train").is_dir() {
            self.images_dir.join("train")
        } else {
            self.images_dir.clone()
        };
        let val_dir = if val_split {
            train_dir.clone()
        } else if self.images_dir.join("val").is_dir() {
            self.images_dir.join("val")
        } else {
            train_dir.clone()
        };

        let mut s = String::new();
        s.push_str("# 由 Forge YOLO 生成\n");
        s.push_str(&format!("path: {}\n", self.root.display()));
        s.push_str(&format!("train: {}\n", rel(&train_dir)));
        s.push_str(&format!("val: {}\n", rel(&val_dir)));
        s.push_str(&format!("nc: {}\n", self.classes.len()));
        s.push_str("names:\n");
        for (i, name) in self.classes.iter().enumerate() {
            s.push_str(&format!("  {i}: {name}\n"));
        }
        s
    }
}

fn collect_images(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let read = std::fs::read_dir(dir).with_context(|| format!("无法读取目录 {}", dir.display()))?;
    for entry in read {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_images(&path, out)?;
        } else if is_image(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// 读取类别名：优先 classes.txt，其次 data.yaml 里的 names。
pub fn load_classes(root: &Path, images_dir: &Path) -> Vec<String> {
    let candidates = [
        root.join("classes.txt"),
        images_dir.join("classes.txt"),
        root.join("classes.names"),
        images_dir.join("classes.names"),
    ];
    for path in candidates {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let names: Vec<String> = text
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            if !names.is_empty() {
                return names;
            }
        }
    }

    for path in [root.join("data.yaml"), root.join("dataset.yaml")] {
        if let Ok(text) = std::fs::read_to_string(&path) {
            let names = parse_yaml_names(&text);
            if !names.is_empty() {
                return names;
            }
        }
    }
    Vec::new()
}

/// 极简 data.yaml 解析，只关心 names 字段。
fn parse_yaml_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_names = false;
    for line in text.lines() {
        let trimmed = line.trim_end();
        let no_comment = match trimmed.find('#') {
            Some(i) => &trimmed[..i],
            None => trimmed,
        };
        let t = no_comment.trim();
        if t.is_empty() {
            continue;
        }

        if !in_names {
            if let Some(rest) = t.strip_prefix("names:") {
                let rest = rest.trim();
                if rest.starts_with('[') {
                    let inner = rest.trim_start_matches('[').trim_end_matches(']');
                    for part in inner.split(',') {
                        let name = part.trim().trim_matches(|c| c == '\'' || c == '"');
                        if !name.is_empty() {
                            names.push(name.to_string());
                        }
                    }
                    return names;
                }
                in_names = true;
            }
            continue;
        }

        if !line.starts_with(' ') && !line.starts_with('\t') {
            break;
        }
        if let Some(rest) = t.strip_prefix("- ") {
            names.push(
                rest.trim()
                    .trim_matches(|c| c == '\'' || c == '"')
                    .to_string(),
            );
        } else if let Some((idx, rest)) = t.split_once(':') {
            if idx.trim().parse::<usize>().is_ok() {
                names.push(
                    rest.trim()
                        .trim_matches(|c| c == '\'' || c == '"')
                        .to_string(),
                );
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_yaml_block_names() {
        let text = "path: /data\nnames:\n  0: cat\n  1: dog\n";
        assert_eq!(parse_yaml_names(text), vec!["cat", "dog"]);
    }

    #[test]
    fn parse_yaml_inline_names() {
        let text = "names: ['cat', 'dog']\n";
        assert_eq!(parse_yaml_names(text), vec!["cat", "dog"]);
    }

    #[test]
    fn parse_yaml_list_names() {
        let text = "names:\n  - cat\n  - dog\nother: 1\n";
        assert_eq!(parse_yaml_names(text), vec!["cat", "dog"]);
    }

    #[test]
    fn image_extension_check() {
        assert!(is_image(Path::new("a.JPG")));
        assert!(is_image(Path::new("a.png")));
        assert!(!is_image(Path::new("a.txt")));
    }
}
