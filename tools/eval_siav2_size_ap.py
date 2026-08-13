import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    box_iou,
    check_dataset,
    check_img_size,
    colorstr,
    increment_path,
    non_max_suppression,
    scale_coords,
    set_logging,
    xywh2xyxy,
)
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, time_synchronized


def bucket_masks(labels, small_max_side, medium_max_side):
    if labels.numel() == 0:
        empty = torch.zeros(0, dtype=torch.bool, device=labels.device)
        return {"small": empty, "medium": empty, "large": empty}
    max_side = labels[:, 3:5].max(1)[0]
    small = max_side <= small_max_side
    medium = (max_side > small_max_side) & (max_side <= medium_max_side)
    large = max_side > medium_max_side
    return {"small": small, "medium": medium, "large": large}


def process_image(pred, labels, img_shape, shape_info, iouv, bucket_mask=None):
    if bucket_mask is not None:
        labels = labels[bucket_mask]
    nl = len(labels)
    tcls = labels[:, 0].tolist() if nl else []
    correct = torch.zeros(pred.shape[0], iouv.numel(), dtype=torch.bool, device=pred.device)
    if pred.shape[0] == 0:
        return correct, pred[:, 4], pred[:, 5], tcls

    predn = pred.clone()
    scale_coords(img_shape, predn[:, :4], shape_info[0], shape_info[1])
    if nl:
        tbox = xywh2xyxy(labels[:, 1:5])
        scale_coords(img_shape, tbox, shape_info[0], shape_info[1])
        tcls_tensor = labels[:, 0]
        detected = []
        for cls in torch.unique(tcls_tensor):
            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
            pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
            if pi.shape[0]:
                ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                detected_set = set()
                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                    d = ti[i[j]]
                    if d.item() not in detected_set:
                        detected_set.add(d.item())
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv
                        if len(detected) == nl:
                            break
    return correct, pred[:, 4], pred[:, 5], tcls


def summarize_stats(stats, nc, v5_metric=False):
    if not stats:
        return empty_summary()
    arrays = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(arrays[3]):
        nt = np.bincount(arrays[3].astype(np.int64), minlength=nc)
    else:
        nt = np.zeros(nc, dtype=np.int64)
    if len(arrays) and len(arrays[0]) and arrays[0].any() and len(arrays[3]):
        p, r, ap, f1, ap_class = ap_per_class(*arrays, v5_metric=v5_metric)
        ap50 = ap[:, 0]
        ap5095 = ap.mean(1)
        return {
            "labels": int(nt.sum()),
            "predictions": int(len(arrays[1])),
            "precision": float(p.mean()) if len(p) else 0.0,
            "recall": float(r.mean()) if len(r) else 0.0,
            "map50": float(ap50.mean()) if len(ap50) else 0.0,
            "map50_95": float(ap5095.mean()) if len(ap5095) else 0.0,
            "ap_class": [int(x) for x in ap_class.tolist()],
            "targets_per_class": [int(x) for x in nt.tolist()],
        }
    return {
        "labels": int(nt.sum()),
        "predictions": int(len(arrays[1])) if len(arrays) > 1 else 0,
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "ap_class": [],
        "targets_per_class": [int(x) for x in nt.tolist()],
    }


def empty_summary():
    return {
        "labels": 0,
        "predictions": 0,
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "ap_class": [],
        "targets_per_class": [],
    }


def write_markdown(path, payload):
    lines = ["# SIAV2 Size AP Evaluation", ""]
    lines.append(f"weights: `{payload['weights']}`")
    lines.append(f"data: `{payload['data']}`")
    lines.append(f"task: `{payload['task']}`")
    lines.append(f"imgsz: `{payload['imgsz']}`")
    lines.append(f"bucket_unit: `{payload['bucket_unit']}`")
    lines.append(f"small_max_side: `{payload['small_max_side']}`")
    lines.append(f"medium_max_side: `{payload['medium_max_side']}`")
    lines.append("")
    lines.append("| bucket | labels | predictions | P | R | mAP50 | mAP50-95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, item in payload["metrics"].items():
        lines.append(
            f"| {name} | {item['labels']} | {item['predictions']} | "
            f"{item['precision']:.5f} | {item['recall']:.5f} | {item['map50']:.5f} | {item['map50_95']:.5f} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def evaluate(opt):
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch_size)
    model = attempt_load(opt.weights, map_location=device)
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.img_size, s=gs)
    half = device.type != "cpu" and not opt.no_half
    if half:
        model.half()
    model.eval()

    with open(opt.data, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    check_dataset(data)
    nc = 1 if opt.single_cls else int(data["nc"])
    task = opt.task if opt.task in ("train", "val", "test") else "val"
    loader_opt = SimpleNamespace(single_cls=opt.single_cls)
    dataloader = create_dataloader(
        data[task],
        imgsz,
        opt.batch_size,
        gs,
        loader_opt,
        pad=0.5,
        rect=True,
        workers=opt.workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    stats = {name: [] for name in ("all", "small", "medium", "large")}
    seen = 0
    t0 = 0.0
    t1 = 0.0
    pbar = tqdm(dataloader, desc="size-ap")
    for img, targets, _paths, shapes in pbar:
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape
        with torch.no_grad():
            t = time_synchronized()
            out, _train_out = model(img)
            t0 += time_synchronized() - t
            targets[:, 2:] *= torch.tensor([width, height, width, height], device=device)
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)
            t1 += time_synchronized() - t

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            masks = bucket_masks(labels, opt.small_max_side, opt.medium_max_side)
            seen += 1
            for bucket_name, mask in [("all", None), ("small", masks["small"]), ("medium", masks["medium"]), ("large", masks["large"])]:
                correct, conf, pred_cls, tcls = process_image(pred, labels, img[si].shape[1:], shapes[si], iouv, mask)
                stats[bucket_name].append((correct.cpu(), conf.cpu(), pred_cls.cpu(), tcls))

    metrics = {name: summarize_stats(bucket_stats, nc, opt.v5_metric) for name, bucket_stats in stats.items()}
    payload = {
        "weights": opt.weights,
        "data": str(Path(opt.data).resolve()),
        "task": task,
        "imgsz": imgsz,
        "batch_size": opt.batch_size,
        "conf_thres": opt.conf_thres,
        "iou_thres": opt.iou_thres,
        "bucket_unit": "input image pixels after dataloader letterbox",
        "small_max_side": opt.small_max_side,
        "medium_max_side": opt.medium_max_side,
        "seen_images": seen,
        "speed_ms": {
            "inference_per_image": t0 / max(seen, 1) * 1e3,
            "nms_per_image": t1 / max(seen, 1) * 1e3,
            "total_per_image": (t0 + t1) / max(seen, 1) * 1e3,
        },
        "metrics": metrics,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv7 AP by target max-side size bucket.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--task", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--v5-metric", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--small-max-side", type=float, default=64.0, help="small bucket max side in input image pixels after letterbox")
    parser.add_argument("--medium-max-side", type=float, default=128.0, help="medium bucket max side in input image pixels after letterbox")
    parser.add_argument("--project", default="runs/siav2_eval")
    parser.add_argument("--name", default="size_ap")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--md-out", default="")
    opt = parser.parse_args()

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate(opt)

    json_out = Path(opt.json_out) if opt.json_out else save_dir / "size_ap.json"
    md_out = Path(opt.md_out) if opt.md_out else save_dir / "size_ap.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
