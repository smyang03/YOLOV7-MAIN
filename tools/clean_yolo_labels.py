import argparse
import shutil
from pathlib import Path

import yaml

from siav2_dataset_eda import expand_images, image_to_label, read_yaml


def clean_label_file(path, nc=None, write=False, backup_dir=None):
    original = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    kept = []
    dropped = []

    for line_no, line in enumerate(original, start=1):
        text = line.strip()
        if not text:
            dropped.append((line_no, "empty", line))
            continue
        parts = text.split()
        if len(parts) != 5:
            dropped.append((line_no, "malformed_columns", line))
            continue
        try:
            cls_raw, x, y, w, h = [float(v) for v in parts]
        except ValueError:
            dropped.append((line_no, "nonnumeric", line))
            continue
        if not cls_raw.is_integer():
            dropped.append((line_no, "noninteger_class", line))
            continue
        cls = int(cls_raw)
        if cls < 0 or (nc is not None and cls >= nc):
            dropped.append((line_no, "class_out_of_range", line))
            continue
        if min(x, y, w, h) < 0.0 or max(x, y, w, h) > 1.0:
            dropped.append((line_no, "coord_out_of_range", line))
            continue
        if w <= 0.0 or h <= 0.0:
            dropped.append((line_no, "nonpositive_wh", line))
            continue
        kept.append(text)

    if write and dropped:
        if backup_dir:
            backup_path = backup_dir / path.relative_to(path.anchor)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    return len(original), len(kept), dropped


def resolve_data_paths(data_path):
    data_path = Path(data_path).resolve()
    data = read_yaml(data_path)
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    roots = [root, data_path.parent]
    nc = int(data["nc"]) if "nc" in data else None
    split_sources = []
    for split in ("train", "val", "test"):
        source = data.get(split)
        if source:
            split_sources.append((split, source))
    return nc, roots, split_sources


def collect_label_paths(data_path=None, label_dirs=None):
    labels = set()
    nc = None
    if data_path:
        nc, roots, split_sources = resolve_data_paths(data_path)
        for _split, source in split_sources:
            for image_path in expand_images(source, roots):
                labels.add(image_to_label(image_path))
    for label_dir in label_dirs or []:
        labels.update(Path(label_dir).resolve().glob("*.txt"))
    return sorted(labels), nc


def main():
    parser = argparse.ArgumentParser(description="Drop invalid YOLO label rows from dataset labels.")
    parser.add_argument("--data", help="YOLO data yaml; train/val/test images are mapped to label files")
    parser.add_argument("--label-dir", action="append", default=[], help="label directory to scan; can be repeated")
    parser.add_argument("--nc", type=int, default=None, help="override class count for label validation")
    parser.add_argument("--write", action="store_true", help="modify label files; otherwise dry-run only")
    parser.add_argument("--backup-dir", type=Path, default=None, help="optional backup root for changed label files")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    if not args.data and not args.label_dir:
        raise SystemExit("provide --data and/or --label-dir")

    label_paths, data_nc = collect_label_paths(args.data, args.label_dir)
    nc = args.nc if args.nc is not None else data_nc
    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    missing = 0
    changed_files = 0
    total_lines = 0
    kept_lines = 0
    dropped_lines = 0
    examples = []
    reasons = {}

    for label_path in label_paths:
        if not label_path.exists():
            missing += 1
            continue
        scanned += 1
        line_count, kept_count, dropped = clean_label_file(
            label_path,
            nc=nc,
            write=args.write,
            backup_dir=args.backup_dir,
        )
        total_lines += line_count
        kept_lines += kept_count
        dropped_lines += len(dropped)
        if dropped:
            changed_files += 1
            for line_no, reason, text in dropped:
                reasons[reason] = reasons.get(reason, 0) + 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "file": str(label_path),
                            "line": line_no,
                            "reason": reason,
                            "text": text,
                        }
                    )

    print(
        yaml.safe_dump(
            {
                "mode": "write" if args.write else "dry_run",
                "nc": nc,
                "label_files": len(label_paths),
                "scanned_files": scanned,
                "missing_files": missing,
                "changed_files": changed_files,
                "total_lines": total_lines,
                "kept_lines": kept_lines,
                "dropped_lines": dropped_lines,
                "drop_reasons": reasons,
                "examples": examples,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )


if __name__ == "__main__":
    main()
