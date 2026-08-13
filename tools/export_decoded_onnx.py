"""Export a YOLOv7 checkpoint with a decoded concatenated detection output."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.yolo import Detect, IAuxDetect, IDetect  # noqa: E402
from models.yolo import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--img', type=int, default=640)
    ap.add_argument('--opset', type=int, default=12)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location='cpu')
    src = ckpt.get('ema') or ckpt.get('model') if isinstance(ckpt, dict) else ckpt
    nc = getattr(src, 'nc', None)
    model = Model(args.cfg, nc=nc).float().to(device).eval()
    source_sd = src.float().state_dict() if hasattr(src, 'state_dict') else src
    current = model.state_dict()
    copied = 0
    for key, value in source_sd.items():
        if key in current and current[key].shape == value.shape:
            current[key] = value
            copied += 1
    model.load_state_dict(current, strict=False)
    with torch.no_grad():
        model.fuse().eval()

    det_types = (Detect, IDetect, IAuxDetect)
    for module in model.modules():
        if isinstance(module, det_types):
            module.export = False
            module.concat = True
            module.include_nms = False
            module.end2end = False

    x = torch.zeros(1, 3, args.img, args.img, device=device, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    print('copied', copied, 'output', tuple(y.shape))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, x, str(out), opset_version=args.opset,
                      input_names=['images'], output_names=['output'],
                      do_constant_folding=True)
    print('saved', out)


if __name__ == '__main__':
    main()
