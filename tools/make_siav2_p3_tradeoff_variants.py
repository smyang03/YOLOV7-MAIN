import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def write_cfg(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False, width=120)


def common_cfg(base, nc, width):
    cfg = deepcopy(base)
    cfg["nc"] = int(nc)
    cfg["width_multiple"] = float(width)
    return cfg


def keep_through(base, global_layer):
    return global_layer - len(base["backbone"]) + 1


def make_p4p5_pruned(base, nc, width):
    cfg = common_cfg(base, nc, width)
    cfg["anchors"] = deepcopy(base["anchors"][1:3])  # P4/P5
    head = deepcopy(base["head"][:keep_through(base, 71)])
    head.extend(
        [
            [71, 1, "Conv", [512, 3, 1]],  # 72 P4 detect feature
            [71, 1, "Conv", [384, 3, 2]],  # 73 P4 -> P5
            [[-1, 59], 1, "Concat", [1]],  # 74
            [-1, 1, "Conv", [384, 1, 1]],  # 75
            [-2, 1, "Conv", [384, 1, 1]],  # 76
            [-1, 1, "Conv", [192, 3, 1]],  # 77
            [-1, 1, "Conv", [192, 3, 1]],  # 78
            [-1, 1, "Conv", [192, 3, 1]],  # 79
            [-1, 1, "Conv", [192, 3, 1]],  # 80
            [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],  # 81
            [-1, 1, "Conv", [384, 1, 1]],  # 82 P5
            [82, 1, "Conv", [768, 3, 1]],  # 83 P5 detect feature
            [[72, 83], 1, "Detect", ["nc", "anchors"]],  # 84 Detect(P4,P5)
        ]
    )
    cfg["head"] = head
    return cfg


def make_p3lite_p4p5(base, nc, width):
    cfg = common_cfg(base, nc, width)
    cfg["anchors"] = deepcopy(base["anchors"][:3])  # P3/P4/P5
    head = deepcopy(base["head"][:keep_through(base, 71)])
    head.extend(
        [
            [71, 1, "Conv", [128, 1, 1]],  # 72 P4 reduce for P3 side branch
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 73
            [19, 1, "Conv", [128, 1, 1]],  # 74 route backbone P3
            [[-1, -2], 1, "Concat", [1]],  # 75
            [-1, 1, "Conv", [128, 1, 1]],  # 76
            [-1, 1, "Conv", [128, 3, 1]],  # 77 P3-lite detect feature
            [71, 1, "Conv", [512, 3, 1]],  # 78 P4 detect feature
            [71, 1, "Conv", [384, 3, 2]],  # 79 P4 -> P5
            [[-1, 59], 1, "Concat", [1]],  # 80
            [-1, 1, "Conv", [384, 1, 1]],  # 81
            [-2, 1, "Conv", [384, 1, 1]],  # 82
            [-1, 1, "Conv", [192, 3, 1]],  # 83
            [-1, 1, "Conv", [192, 3, 1]],  # 84
            [-1, 1, "Conv", [192, 3, 1]],  # 85
            [-1, 1, "Conv", [192, 3, 1]],  # 86
            [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],  # 87
            [-1, 1, "Conv", [384, 1, 1]],  # 88 P5
            [88, 1, "Conv", [768, 3, 1]],  # 89 P5 detect feature
            [[77, 78, 89], 1, "Detect", ["nc", "anchors"]],  # 90 Detect(P3,P4,P5)
        ]
    )
    cfg["head"] = head
    return cfg


def make_p3lite_p4p6(base, nc, width):
    cfg = common_cfg(base, nc, width)
    cfg["anchors"] = deepcopy(base["anchors"])  # P3/P4/P5/P6
    head = deepcopy(base["head"][:keep_through(base, 71)])
    head.extend(
        [
            [71, 1, "Conv", [128, 1, 1]],  # 72 P4 reduce for P3 side branch
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],  # 73
            [19, 1, "Conv", [128, 1, 1]],  # 74 route backbone P3
            [[-1, -2], 1, "Concat", [1]],  # 75
            [-1, 1, "Conv", [128, 1, 1]],  # 76
            [-1, 1, "Conv", [128, 3, 1]],  # 77 P3-lite detect feature
            [71, 1, "Conv", [512, 3, 1]],  # 78 P4 detect feature
            [71, 1, "Conv", [384, 3, 2]],  # 79 P4 -> P5
            [[-1, 59], 1, "Concat", [1]],  # 80
            [-1, 1, "Conv", [384, 1, 1]],  # 81
            [-2, 1, "Conv", [384, 1, 1]],  # 82
            [-1, 1, "Conv", [192, 3, 1]],  # 83
            [-1, 1, "Conv", [192, 3, 1]],  # 84
            [-1, 1, "Conv", [192, 3, 1]],  # 85
            [-1, 1, "Conv", [192, 3, 1]],  # 86
            [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],  # 87
            [-1, 1, "Conv", [384, 1, 1]],  # 88 P5
            [-1, 1, "Conv", [512, 3, 2]],  # 89 P5 -> P6
            [[-1, 47], 1, "Concat", [1]],  # 90
            [-1, 1, "Conv", [512, 1, 1]],  # 91
            [-2, 1, "Conv", [512, 1, 1]],  # 92
            [-1, 1, "Conv", [256, 3, 1]],  # 93
            [-1, 1, "Conv", [256, 3, 1]],  # 94
            [-1, 1, "Conv", [256, 3, 1]],  # 95
            [-1, 1, "Conv", [256, 3, 1]],  # 96
            [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],  # 97
            [-1, 1, "Conv", [512, 1, 1]],  # 98 P6
            [88, 1, "Conv", [768, 3, 1]],  # 99 P5 detect feature
            [98, 1, "Conv", [1024, 3, 1]],  # 100 P6 detect feature
            [[77, 78, 99, 100], 1, "Detect", ["nc", "anchors"]],  # 101 Detect(P3,P4,P5,P6)
        ]
    )
    cfg["head"] = head
    return cfg


def make_p3full_p4p5(base, nc, width):
    cfg = common_cfg(base, nc, width)
    cfg["anchors"] = deepcopy(base["anchors"][:3])  # P3/P4/P5
    head = deepcopy(base["head"][:keep_through(base, 103)])
    head.extend(
        [
            [83, 1, "Conv", [256, 3, 1]],  # 104 P3 detect feature
            [93, 1, "Conv", [512, 3, 1]],  # 105 P4 detect feature
            [103, 1, "Conv", [768, 3, 1]],  # 106 P5 detect feature
            [[104, 105, 106], 1, "Detect", ["nc", "anchors"]],  # 107 Detect(P3,P4,P5)
        ]
    )
    cfg["head"] = head
    return cfg


def make_training_p3lite_p4p5(base, nc, width):
    cfg = make_p3lite_p4p5(base, nc, width)
    head = deepcopy(cfg["head"][:-1])
    head.extend(
        [
            [76, 1, "Conv", [256, 3, 1]],  # 90 P3 aux
            [71, 1, "Conv", [640, 3, 1]],  # 91 P4 aux
            [59, 1, "Conv", [960, 3, 1]],  # 92 P5 aux
            [[77, 78, 89, 90, 91, 92], 1, "IAuxDetect", ["nc", "anchors"]],  # 93 Detect(P3,P4,P5)
        ]
    )
    cfg["head"] = head
    return cfg


def make_training_p3lite_p4p6(base, nc, width):
    cfg = make_p3lite_p4p6(base, nc, width)
    head = deepcopy(cfg["head"][:-1])
    head.extend(
        [
            [76, 1, "Conv", [256, 3, 1]],  # 101 P3 aux
            [71, 1, "Conv", [640, 3, 1]],  # 102 P4 aux
            [59, 1, "Conv", [960, 3, 1]],  # 103 P5 aux
            [47, 1, "Conv", [1280, 3, 1]],  # 104 P6 aux
            [[77, 78, 99, 100, 101, 102, 103, 104], 1, "IAuxDetect", ["nc", "anchors"]],  # 105 Detect(P3,P4,P5,P6)
        ]
    )
    cfg["head"] = head
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Create SIAV2 P3/deep-head tradeoff deploy cfgs.")
    parser.add_argument("--base", default="cfg/deploy/yolov7-w6.yaml")
    parser.add_argument("--out-dir", default="cfg/deploy")
    parser.add_argument("--training-out-dir", default="cfg/training")
    parser.add_argument("--nc", type=int, default=16)
    parser.add_argument("--width", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    variants = {
        "yolov7-l6-siav2-p4p5-pruned-w250.yaml": make_p4p5_pruned(base, args.nc, args.width),
        "yolov7-l6-siav2-p3lite-p4p6-w250.yaml": make_p3lite_p4p6(base, args.nc, args.width),
        "yolov7-l6-siav2-p3lite-p4p5-w250.yaml": make_p3lite_p4p5(base, args.nc, args.width),
        "yolov7-l6-siav2-p3full-p4p5-w250.yaml": make_p3full_p4p5(base, args.nc, args.width),
    }
    for name, cfg in variants.items():
        path = out_dir / name
        write_cfg(path, cfg)
        print(path)

    if args.training_out_dir:
        training_path = Path(args.training_out_dir) / "yolov7-l6-siav2-p3lite-p4p5-w250.yaml"
        write_cfg(training_path, make_training_p3lite_p4p5(base, args.nc, args.width))
        print(training_path)
        training_path = Path(args.training_out_dir) / "yolov7-l6-siav2-p3lite-p4p6-w250.yaml"
        write_cfg(training_path, make_training_p3lite_p4p6(base, args.nc, args.width))
        print(training_path)


if __name__ == "__main__":
    main()
