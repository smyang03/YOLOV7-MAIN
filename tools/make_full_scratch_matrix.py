"""Generate the complete scratch-training experiment matrix."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import yaml

from make_siav2_p3_tradeoff_variants import make_training_p3lite_p4p5, make_training_p3lite_p4p6


def write_cfg(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False, width=120)


def customize(cfg, p3_channels, width, p3_depth=1):
    cfg = deepcopy(cfg)
    cfg["width_multiple"] = float(width)
    backbone_len = len(cfg["backbone"])
    for local_index, layer in enumerate(cfg["head"]):
        global_index = backbone_len + local_index
        if global_index in {72, 74, 76, 77} and layer[2] == "Conv":
            args = list(layer[3])
            args[0] = int(p3_channels)
            layer[3] = args
            if global_index == 77:
                layer[1] = int(p3_depth)
    return cfg


def main():
    parser = ArgumentParser(description="Generate full scratch experiment configs.")
    parser.add_argument("--base", default="cfg/deploy/yolov7-w6.yaml")
    parser.add_argument("--out-dir", default="cfg/training")
    parser.add_argument("--nc", type=int, default=16)
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    matrix = [
        ("s01_p3lite_p4p5_w025", "p4p5", 128, 0.25, 1),
        ("s02_p3w150_p4p5_w025", "p4p5", 192, 0.25, 1),
        ("s03_p3w200_p4p5_w025", "p4p5", 256, 0.25, 1),
        ("s04_p3lite_p4p6_w025", "p4p6", 128, 0.25, 1),
        ("s05_p3w150_p4p6_w025", "p4p6", 192, 0.25, 1),
        ("s06_p3w200_p4p6_w025", "p4p6", 256, 0.25, 1),
        ("s07_p3w200_p4p5_w031", "p4p5", 256, 0.3125, 1),
        ("s08_p3w200_p4p6_w031", "p4p6", 256, 0.3125, 1),
        ("s09_p3w200_p4p6_w0375", "p4p6", 256, 0.375, 1),
        ("s10_p3w200_p4p6_w0375_deep", "p4p6", 256, 0.375, 2),
        ("s11_p3w200_p4p6_w0375_small", "p4p6", 256, 0.375, 1),
        ("s12_p3w200_p4p6_w050", "p4p6", 256, 0.50, 1),
    ]
    for name, family, p3_channels, width, p3_depth in matrix:
        cfg = (make_training_p3lite_p4p5 if family == "p4p5" else make_training_p3lite_p4p6)(base, args.nc, width)
        write_cfg(Path(args.out_dir) / f"{name}.yaml", customize(cfg, p3_channels, width, p3_depth))
        print(Path(args.out_dir) / f"{name}.yaml")


if __name__ == "__main__":
    main()
