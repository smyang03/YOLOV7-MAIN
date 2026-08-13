"""Create P3 channel-recovery training/deploy configs.

The existing p3lite candidate uses 128-channel P3 branches before
width_multiple is applied.  At width_multiple=0.25 this becomes 32 channels,
which is a likely bottleneck for small-object recall.  This tool keeps the
backbone, P4/P5 path, anchors and loss-compatible auxiliary head unchanged and
only widens the P3 branch.
"""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import yaml

from make_siav2_p3_tradeoff_variants import (
    make_p3lite_p4p5,
    make_training_p3lite_p4p5,
)


def write_cfg(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False, width=120)


def widen_p3(cfg, p3_channels):
    """Widen P3-only layers by their global layer index.

    Global indices are stable because this function is applied after the
    existing p3lite graph has been generated.  Training configs additionally
    contain auxiliary layers, which do not need widening.
    """
    cfg = deepcopy(cfg)
    backbone_len = len(cfg["backbone"])
    p3_global_layers = {72, 74, 76, 77}

    for local_index, layer in enumerate(cfg["head"]):
        global_index = backbone_len + local_index
        if global_index in p3_global_layers and layer[2] == "Conv":
            args = list(layer[3])
            args[0] = int(p3_channels)
            layer[3] = args
    return cfg


def main():
    parser = ArgumentParser(description="Generate P3 channel recovery variants.")
    parser.add_argument("--base", default="cfg/deploy/yolov7-w6.yaml")
    parser.add_argument("--deploy-out-dir", default="cfg/deploy")
    parser.add_argument("--training-out-dir", default="cfg/training")
    parser.add_argument("--nc", type=int, default=16)
    parser.add_argument("--width", type=float, default=0.25)
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    variants = {
        "p3w150": 192,
        "p3w200": 256,
    }
    for name, channels in variants.items():
        deploy = widen_p3(make_p3lite_p4p5(base, args.nc, args.width), channels)
        training = widen_p3(make_training_p3lite_p4p5(base, args.nc, args.width), channels)
        write_cfg(Path(args.deploy_out_dir) / f"yolov7-l6-siav2-p3lite-{name}-w250.yaml", deploy)
        write_cfg(Path(args.training_out_dir) / f"yolov7-l6-siav2-p3lite-{name}-w250.yaml", training)


if __name__ == "__main__":
    main()
