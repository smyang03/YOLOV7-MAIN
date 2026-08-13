import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def resolve_existing_path(path, roots):
    path = Path(path)
    if path.is_absolute():
        return path
    for root in roots:
        candidate = (Path(root) / path).resolve()
        if candidate.exists():
            return candidate
    return (Path(roots[0]) / path).resolve()


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_to_label(path):
    parts = list(path.parts)
    for image_dir in ("images", "JPEGImages"):
        if image_dir in parts:
            idx = len(parts) - 1 - parts[::-1].index(image_dir)
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return path.with_suffix(".txt")


def expand_images(source, roots):
    if isinstance(source, (list, tuple)):
        images = []
        for item in source:
            images.extend(expand_images(item, roots))
        return images

    path = resolve_existing_path(str(source), roots)
    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img = Path(line)
            if img.is_absolute():
                images.append(img)
            else:
                list_root = path.parent
                images.append(resolve_existing_path(img, [list_root, *roots]))
        return images
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if path.is_file() and path.suffix.lower() in IMG_EXTS:
        return [path]
    return []


def add_error(errors, error_type, path, line_no=None, message="", max_examples=20):
    errors[error_type] += 1
    if len(errors["examples"]) < max_examples:
        entry = {
            "type": error_type,
            "file": str(path),
        }
        if line_no is not None:
            entry["line"] = int(line_no)
        if message:
            entry["message"] = message
        errors["examples"].append(entry)


def read_label(path, nc=None, max_examples=20):
    rows = []
    errors = Counter()
    errors["examples"] = []
    if not path.exists():
        return rows, False, errors

    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            add_error(errors, "malformed_columns", path, line_no, f"expected 5 columns, got {len(parts)}", max_examples)
            continue
        try:
            cls_raw, x, y, w, h = [float(v) for v in parts]
        except ValueError:
            add_error(errors, "nonnumeric", path, line_no, text, max_examples)
            continue
        if not cls_raw.is_integer():
            add_error(errors, "noninteger_class", path, line_no, str(cls_raw), max_examples)
            continue
        cls = int(cls_raw)
        if cls < 0:
            add_error(errors, "negative_class", path, line_no, str(cls), max_examples)
            continue
        if nc is not None and cls >= nc:
            add_error(errors, "class_out_of_range", path, line_no, f"class {cls} >= nc {nc}", max_examples)
            continue
        if min(x, y, w, h) < 0.0 or max(x, y, w, h) > 1.0:
            add_error(errors, "coord_out_of_range", path, line_no, f"{x} {y} {w} {h}", max_examples)
            continue
        if w <= 0.0 or h <= 0.0:
            add_error(errors, "nonpositive_wh", path, line_no, f"{w} {h}", max_examples)
            continue
        row = (cls, x, y, w, h)
        if row in seen:
            add_error(errors, "duplicate_label", path, line_no, text, max_examples)
            continue
        seen.add(row)
        rows.append(row)
    return rows, True, errors


def load_anchors(cfg_path):
    if not cfg_path:
        return []
    cfg = read_yaml(cfg_path)
    anchors = cfg.get("anchors", [])
    flat = []
    for layer in anchors:
        vals = list(layer)
        flat.extend((float(vals[i]), float(vals[i + 1])) for i in range(0, len(vals), 2))
    return flat


def anchor_ratio(wh, anchors):
    if not anchors:
        return None
    w, h = wh
    best = None
    for aw, ah in anchors:
        ratio = max(w / aw, aw / max(w, 1e-9), h / ah, ah / max(h, 1e-9))
        best = ratio if best is None else min(best, ratio)
    return best


def pct(value, total):
    return round(value / total * 100.0, 3) if total else 0.0


def merge_errors(dst, src):
    for key, value in src.items():
        if key == "examples":
            continue
        dst[key] += value
    dst["examples"].extend(src.get("examples", []))


def analyze_split(name, source, roots, imgsz, small_thresholds, anchors, anchor_t, nc, max_examples):
    images = expand_images(source, roots)
    class_counts = Counter()
    side_bins = Counter()
    area_bins = Counter()
    anchor_hits = 0
    anchor_ratios = []
    missing_labels = 0
    empty_labels = 0
    invalid_label_files = 0
    errors = Counter()
    errors["examples"] = []
    boxes = []

    for img in images:
        label_path = image_to_label(img)
        labels, exists, label_errors = read_label(label_path, nc=nc, max_examples=max_examples)
        if not exists:
            missing_labels += 1
        if exists and not labels:
            empty_labels += 1
        label_error_count = sum(v for k, v in label_errors.items() if k != "examples")
        if label_error_count:
            invalid_label_files += 1
            merge_errors(errors, label_errors)
        for cls, _x, _y, w, h in labels:
            max_side_px = max(w, h) * imgsz
            area_px = w * h * imgsz * imgsz
            class_counts[cls] += 1
            boxes.append((cls, max_side_px, area_px))
            for threshold in small_thresholds:
                if max_side_px <= threshold:
                    side_bins[str(threshold)] += 1
            for threshold in (32 * 32, 64 * 64, 96 * 96, 128 * 128):
                if area_px <= threshold:
                    area_bins[str(threshold)] += 1
            ratio = anchor_ratio((w * imgsz, h * imgsz), anchors)
            if ratio is not None:
                anchor_ratios.append(ratio)
                if ratio < anchor_t:
                    anchor_hits += 1

    box_count = len(boxes)
    invalid_label_lines = sum(v for k, v in errors.items() if k != "examples")
    summary = {
        "split": name,
        "images": len(images),
        "boxes": box_count,
        "missing_label_files": missing_labels,
        "empty_label_files": empty_labels,
        "invalid_label_files": invalid_label_files,
        "invalid_label_lines": invalid_label_lines,
        "label_errors": {k: int(v) for k, v in sorted(errors.items()) if k != "examples"},
        "label_error_examples": errors["examples"][:max_examples],
        "boxes_per_image": round(box_count / len(images), 4) if images else 0.0,
        "class_counts": dict(sorted(class_counts.items())),
        "max_side_px_le": {k: {"count": v, "pct": pct(v, box_count)} for k, v in sorted(side_bins.items(), key=lambda x: float(x[0]))},
        "area_px_le": {k: {"count": v, "pct": pct(v, box_count)} for k, v in sorted(area_bins.items(), key=lambda x: float(x[0]))},
    }
    if anchor_ratios:
        sorted_ratios = sorted(anchor_ratios)
        summary["anchor_fit"] = {
            "anchor_t": anchor_t,
            "best_possible_recall_pct": pct(anchor_hits, box_count),
            "ratio_p50": round(sorted_ratios[int(0.50 * (len(sorted_ratios) - 1))], 4),
            "ratio_p90": round(sorted_ratios[int(0.90 * (len(sorted_ratios) - 1))], 4),
            "ratio_p95": round(sorted_ratios[int(0.95 * (len(sorted_ratios) - 1))], 4),
        }
    return summary


def total_invalid(payload, include_missing=False):
    total = 0
    for split in payload["splits"]:
        if include_missing:
            total += split["missing_label_files"]
        total += split["invalid_label_lines"]
    return total


def write_markdown(path, payload):
    lines = ["# SIAV2 Dataset EDA", ""]
    lines.append(f"data: `{payload['data']}`")
    lines.append(f"imgsz: `{payload['imgsz']}`")
    lines.append(f"nc: `{payload.get('nc')}`")
    lines.append(f"invalid label gate count: `{total_invalid(payload)}`")
    lines.append("")
    for split in payload["splits"]:
        lines.append(f"## {split['split']}")
        lines.append("")
        lines.append(f"- images: `{split['images']}`")
        lines.append(f"- boxes: `{split['boxes']}`")
        lines.append(f"- boxes/image: `{split['boxes_per_image']}`")
        lines.append(f"- missing label files: `{split['missing_label_files']}`")
        lines.append(f"- empty label files: `{split['empty_label_files']}`")
        lines.append(f"- invalid label files: `{split['invalid_label_files']}`")
        lines.append(f"- invalid label lines: `{split['invalid_label_lines']}`")
        if split.get("anchor_fit"):
            fit = split["anchor_fit"]
            lines.append(f"- anchor BPR@{fit['anchor_t']}: `{fit['best_possible_recall_pct']}%`")
            lines.append(f"- anchor ratio p50/p90/p95: `{fit['ratio_p50']}` / `{fit['ratio_p90']}` / `{fit['ratio_p95']}`")
        if split["label_errors"]:
            lines.append("")
            lines.append("| label error | count |")
            lines.append("|---|---:|")
            for error_type, count in split["label_errors"].items():
                lines.append(f"| {error_type} | {count} |")
        if split["label_error_examples"]:
            lines.append("")
            lines.append("| example | file | line | message |")
            lines.append("|---|---|---:|---|")
            for item in split["label_error_examples"]:
                lines.append(
                    f"| {item['type']} | `{item['file']}` | {item.get('line', '')} | {item.get('message', '')} |"
                )
        lines.append("")
        lines.append("| max side px <= | count | pct |")
        lines.append("|---:|---:|---:|")
        for threshold, item in split["max_side_px_le"].items():
            lines.append(f"| {threshold} | {item['count']} | {item['pct']} |")
        lines.append("")
        lines.append("| class | boxes |")
        lines.append("|---:|---:|")
        for cls, count in split["class_counts"].items():
            lines.append(f"| {cls} | {count} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize and validate SIAV2 YOLO label distribution and anchor fit.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--cfg", default="", help="optional cfg yaml for anchor fit")
    parser.add_argument("--anchor-t", type=float, default=4.0)
    parser.add_argument("--small-thresholds", nargs="*", type=float, default=[32, 64, 96, 128])
    parser.add_argument("--json-out", default="runs/siav2/eda/siav2_dataset_eda.json")
    parser.add_argument("--md-out", default="runs/siav2/eda/siav2_dataset_eda.md")
    parser.add_argument("--max-error-examples", type=int, default=20)
    parser.add_argument("--fail-on-invalid", action="store_true", help="exit nonzero when invalid label lines are found")
    parser.add_argument("--fail-on-missing-labels", action="store_true", help="count missing label files as invalid for --fail-on-invalid")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    data = read_yaml(data_path)
    yaml_dir = data_path.parent
    data_root = resolve_existing_path(data.get("path", "."), [yaml_dir, Path.cwd()]) if isinstance(data, dict) else yaml_dir
    roots = [data_root, yaml_dir, Path.cwd()]
    nc = int(data["nc"]) if "nc" in data else None
    anchors = load_anchors(args.cfg)
    payload = {
        "data": str(data_path),
        "data_root": str(data_root),
        "imgsz": args.imgsz,
        "nc": nc,
        "names": data.get("names", []),
        "cfg": args.cfg,
        "anchors": anchors,
        "splits": [],
    }
    for split_name in ("train", "val", "test"):
        if split_name in data and data[split_name]:
            payload["splits"].append(
                analyze_split(
                    split_name,
                    data[split_name],
                    roots,
                    args.imgsz,
                    args.small_thresholds,
                    anchors,
                    args.anchor_t,
                    nc,
                    args.max_error_examples,
                )
            )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(md_out, payload)
    print(json.dumps(payload, indent=2))
    if args.fail_on_invalid and total_invalid(payload, include_missing=args.fail_on_missing_labels):
        sys.exit(2)


if __name__ == "__main__":
    main()
