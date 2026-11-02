//! 数据集格式转换：YOLO → （YOLO 划分 / COCO / VOC）。

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::json;

use super::dataset::{Dataset, ImageEntry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetFormat {
    YoloSplit,
    Coco,
    Voc,
}

impl TargetFormat {
    pub const ALL: [Self; 3] = [Self::YoloSplit, Self::Coco, Self::Voc];

    pub fn label(self) -> &'static str {
        match self {
            Self::YoloSplit => "YOLO（划分 train/val + data.yaml）",
            Self::Coco => "COCO（instances_*.json）",
            Self::Voc => "VOC（XML + ImageSets）",
        }
    }

    pub fn folder_name(self) -> &'static str {
        match self {
            Self::YoloSplit => "yolo",
            Self::Coco => "coco",
            Self::Voc => "voc",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub format: TargetFormat,
    /// 验证集比例，0 表示不划分
    pub val_ratio: f32,
    pub copy_images: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            format: TargetFormat::YoloSplit,
            val_ratio: 0.2,
            copy_images: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConvertReport {
    pub images: usize,
    pub train: usize,
    pub val: usize,
    pub boxes: usize,
    pub output: PathBuf,
    pub notes: Vec<String>,
}

impl fmt::Display for ConvertReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "完成：{} 张图片，{} 个框（train {} / val {}），输出到 {}",
            self.images,
            self.boxes,
            self.train,
            self.val,
            self.output.display()
        )
    }
}

fn is_val(index: usize, total: usize, ratio: f32) -> bool {
    if ratio <= 0.0 || total < 2 {
        return false;
    }
    let ratio = ratio.clamp(0.0, 0.9);
    let step = (1.0 / ratio).round().max(2.0) as usize;
    // 保证第一张进 train，后续每 step 张取一张做 val
    index > 0 && index % step == 0
}

pub fn convert(
    dataset: &Dataset,
    out_dir: &Path,
    opts: &ConvertOptions,
    progress: &mut dyn FnMut(String),
) -> Result<ConvertReport> {
    match opts.format {
        TargetFormat::YoloSplit => convert_yolo_split(dataset, out_dir, opts, progress),
        TargetFormat::Coco => convert_coco(dataset, out_dir, opts, progress),
        TargetFormat::Voc => convert_voc(dataset, out_dir, opts, progress),
    }
}

fn image_size(entry: &ImageEntry) -> Result<(u32, u32)> {
    ::image::image_dimensions(&entry.image)
        .with_context(|| format!("读取图片尺寸失败：{}", entry.image.display()))
}

fn copy_image(entry: &ImageEntry, dst_dir: &Path) -> Result<PathBuf> {
    let name = entry
        .image
        .file_name()
        .context("图片没有文件名")?
        .to_owned();
    let dst = dst_dir.join(name);
    std::fs::copy(&entry.image, &dst)
        .with_context(|| format!("复制图片失败：{}", entry.image.display()))?;
    Ok(dst)
}

fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content).with_context(|| format!("写入失败：{}", path.display()))
}

fn convert_yolo_split(
    dataset: &Dataset,
    out_dir: &Path,
    opts: &ConvertOptions,
    progress: &mut dyn FnMut(String),
) -> Result<ConvertReport> {
    let mut report = ConvertReport {
        images: dataset.entries.len(),
        output: out_dir.to_path_buf(),
        ..Default::default()
    };
    let total = dataset.entries.len();

    for (i, entry) in dataset.entries.iter().enumerate() {
        let boxes = entry.read_boxes();
        report.boxes += boxes.len();
        let split = if is_val(i, total, opts.val_ratio) {
            report.val += 1;
            "val"
        } else {
            report.train += 1;
            "train"
        };

        if opts.copy_images {
            copy_image(entry, &out_dir.join("images").join(split))?;
        }
        let label_name = entry
            .image
            .file_stem()
            .context("图片没有文件名")?
            .to_owned();
        let label_path = out_dir
            .join("labels")
            .join(split)
            .join(label_name)
            .with_extension("txt");
        write_file(&label_path, &super::label::to_text(&boxes))?;

        if i % 50 == 0 {
            progress(format!("处理中 {}/{}", i + 1, total));
        }
    }

    let mut yaml = String::from("# 由 Forge YOLO 生成\n");
    yaml.push_str(&format!("path: {}\n", out_dir.display()));
    yaml.push_str("train: images/train\n");
    yaml.push_str("val: images/val\n");
    yaml.push_str("names:\n");
    for (i, name) in dataset.classes.iter().enumerate() {
        yaml.push_str(&format!("  {i}: {name}\n"));
    }
    write_file(&out_dir.join("data.yaml"), &yaml)?;
    write_file(
        &out_dir.join("classes.txt"),
        &dataset.classes.join("\n").to_string(),
    )?;
    report
        .notes
        .push("生成 data.yaml、classes.txt、images/{train,val}、labels/{train,val}".to_string());
    progress(report.to_string());
    Ok(report)
}

fn convert_coco(
    dataset: &Dataset,
    out_dir: &Path,
    opts: &ConvertOptions,
    progress: &mut dyn FnMut(String),
) -> Result<ConvertReport> {
    let mut report = ConvertReport {
        images: dataset.entries.len(),
        output: out_dir.to_path_buf(),
        ..Default::default()
    };
    let total = dataset.entries.len();
    let images_dir = out_dir.join("images");
    std::fs::create_dir_all(&images_dir)?;

    let mut splits = [
        (
            "train",
            Vec::<serde_json::Value>::new(),
            Vec::<serde_json::Value>::new(),
        ),
        (
            "val",
            Vec::<serde_json::Value>::new(),
            Vec::<serde_json::Value>::new(),
        ),
    ];
    let categories: Vec<serde_json::Value> = dataset
        .classes
        .iter()
        .enumerate()
        .map(|(i, name)| json!({"id": i + 1, "name": name, "supercategory": "none"}))
        .collect();

    let mut image_id = 1u64;
    let mut ann_id = 1u64;

    for (i, entry) in dataset.entries.iter().enumerate() {
        let split_idx = usize::from(is_val(i, total, opts.val_ratio));
        if split_idx == 1 {
            report.val += 1;
        } else {
            report.train += 1;
        }
        let boxes = entry.read_boxes();
        report.boxes += boxes.len();

        let (w, h) = image_size(entry)?;
        let dst = if opts.copy_images {
            copy_image(entry, &images_dir)?
        } else {
            entry.image.clone()
        };
        let file_name = dst
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        splits[split_idx].1.push(json!({
            "id": image_id,
            "file_name": file_name,
            "width": w,
            "height": h,
        }));

        for b in &boxes {
            let (x1, y1, x2, y2) = b.corners();
            let bw = (x2 - x1) * w as f32;
            let bh = (y2 - y1) * h as f32;
            splits[split_idx].2.push(json!({
                "id": ann_id,
                "image_id": image_id,
                "category_id": b.class_id + 1,
                "bbox": [x1 * w as f32, y1 * h as f32, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            }));
            ann_id += 1;
        }
        image_id += 1;

        if i % 50 == 0 {
            progress(format!("处理中 {}/{}", i + 1, total));
        }
    }

    for (name, images, annotations) in splits {
        let doc = json!({
            "info": {"description": "Forge YOLO export", "version": "1.0"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        });
        let path = out_dir
            .join("annotations")
            .join(format!("instances_{name}.json"));
        write_file(&path, &serde_json::to_string_pretty(&doc)?)?;
    }
    if !opts.copy_images {
        report
            .notes
            .push("未复制图片，JSON 中的 file_name 为原图文件名".to_string());
    }
    write_file(
        &out_dir.join("classes.txt"),
        &dataset.classes.join("\n").to_string(),
    )?;
    report
        .notes
        .push("生成 annotations/instances_train.json、instances_val.json".to_string());
    progress(report.to_string());
    Ok(report)
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn convert_voc(
    dataset: &Dataset,
    out_dir: &Path,
    opts: &ConvertOptions,
    progress: &mut dyn FnMut(String),
) -> Result<ConvertReport> {
    let mut report = ConvertReport {
        images: dataset.entries.len(),
        output: out_dir.to_path_buf(),
        ..Default::default()
    };
    let total = dataset.entries.len();
    let jpeg_dir = out_dir.join("JPEGImages");
    let ann_dir = out_dir.join("Annotations");
    let sets_dir = out_dir.join("ImageSets").join("Main");
    let mut train_list = Vec::new();
    let mut val_list = Vec::new();

    for (i, entry) in dataset.entries.iter().enumerate() {
        let boxes = entry.read_boxes();
        report.boxes += boxes.len();
        let (w, h) = image_size(entry)?;
        if is_val(i, total, opts.val_ratio) {
            report.val += 1;
        } else {
            report.train += 1;
        }

        let dst = if opts.copy_images {
            copy_image(entry, &jpeg_dir)?
        } else {
            entry.image.clone()
        };
        let file_name = dst
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let stem = dst
            .file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut xml = String::new();
        xml.push_str("<annotation>\n");
        xml.push_str("  <folder>JPEGImages</folder>\n");
        xml.push_str(&format!(
            "  <filename>{}</filename>\n",
            xml_escape(&file_name)
        ));
        xml.push_str("  <size>\n");
        xml.push_str(&format!("    <width>{w}</width>\n"));
        xml.push_str(&format!("    <height>{h}</height>\n"));
        xml.push_str("    <depth>3</depth>\n");
        xml.push_str("  </size>\n");
        for b in &boxes {
            let (x1, y1, x2, y2) = b.corners();
            xml.push_str("  <object>\n");
            xml.push_str(&format!(
                "    <name>{}</name>\n",
                xml_escape(&dataset.class_name(b.class_id))
            ));
            xml.push_str("    <pose>Unspecified</pose>\n");
            xml.push_str("    <truncated>0</truncated>\n");
            xml.push_str("    <difficult>0</difficult>\n");
            xml.push_str("    <bndbox>\n");
            xml.push_str(&format!("      <xmin>{:.0}</xmin>\n", x1 * w as f32));
            xml.push_str(&format!("      <ymin>{:.0}</ymin>\n", y1 * h as f32));
            xml.push_str(&format!("      <xmax>{:.0}</xmax>\n", x2 * w as f32));
            xml.push_str(&format!("      <ymax>{:.0}</ymax>\n", y2 * h as f32));
            xml.push_str("    </bndbox>\n");
            xml.push_str("  </object>\n");
        }
        xml.push_str("</annotation>\n");
        write_file(&ann_dir.join(format!("{stem}.xml")), &xml)?;

        if is_val(i, total, opts.val_ratio) {
            val_list.push(stem);
        } else {
            train_list.push(stem);
        }

        if i % 50 == 0 {
            progress(format!("处理中 {}/{}", i + 1, total));
        }
    }

    write_file(&sets_dir.join("train.txt"), &train_list.join("\n"))?;
    write_file(&sets_dir.join("val.txt"), &val_list.join("\n"))?;
    write_file(
        &out_dir.join("classes.txt"),
        &dataset.classes.join("\n").to_string(),
    )?;
    report
        .notes
        .push("生成 JPEGImages/、Annotations/、ImageSets/Main/{train,val}.txt".to_string());
    progress(report.to_string());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn val_split_is_deterministic() {
        assert!(!is_val(0, 10, 0.2));
        assert!(is_val(5, 10, 0.2));
        assert!(!is_val(4, 10, 0.2));
        assert!(!is_val(3, 10, 0.0));
    }

    #[test]
    fn xml_escape_works() {
        assert_eq!(xml_escape("a&b<c>"), "a&amp;b&lt;c&gt;");
    }
}
