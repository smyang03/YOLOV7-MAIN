"""
Phase 0 (속도 게이트 G0) 유틸 — SIAV2_DESIGN.md 참조.

학습 없이 latency 를 측정하기 위해, cfg 로부터 '랜덤 초기화' 모델을 만들어
export.py / attempt_load 가 그대로 소비할 수 있는 .pt 체크포인트로 저장한다.

latency 는 weight 값과 무관하므로, 이 랜덤 weight 로 뽑은 ONNX/TRT latency 는
학습된 모델의 latency 와 동일하다. => 학습 전에 속도 목표 도달 여부를 판정할 수 있다.

width 스윕(0.5 / 0.625 / 0.75)은 파일 복제 없이 --width 로 override 한다.

사용 예:
    # width 0.5 로 랜덤 weight 생성
    python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.5 --nc 80
    # -> yolov7-l6-siav2-w500-random.pt

    # 이후 동일 조건 export & TRT (SIAV2_DESIGN.md Phase 0 공정성 규정 준수)
    python export.py --weights yolov7-l6-siav2-w500-random.pt --img-size 1280 1280 --grid --end2end --simplify
    trtexec --onnx=yolov7-l6-siav2-w500-random.onnx --fp16 --shapes=images:1x3x1280x1280
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml
import torch

# repo 루트를 import path 에 추가 (tools/ 하위에서 실행돼도 동작)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description='SIAV2 Phase 0: build random-init weights for latency profiling')
    p.add_argument('--cfg', required=True, help='model cfg yaml (예: cfg/training/yolov7-l6-siav2.yaml)')
    p.add_argument('--nc', type=int, default=None, help='num classes override (siav2 실제 클래스 수)')
    p.add_argument('--width', type=float, default=None, help='width_multiple override (스윕: 0.5 / 0.625 / 0.75)')
    p.add_argument('--out', default=None, help='출력 .pt 경로 (미지정 시 자동 명명)')
    p.add_argument('--half', action='store_true', help='fp16 로 저장 (train.py 체크포인트와 동일 형식)')
    a = p.parse_args()

    with open(a.cfg) as f:
        d = yaml.load(f, Loader=yaml.SafeLoader)
    if a.width is not None:
        d['width_multiple'] = float(a.width)

    # Model 은 cfg dict 를 직접 받는다. nc override 는 Model 내부에서 처리.
    model = Model(deepcopy(d), ch=3, nc=a.nc)
    model.eval()

    nc = a.nc if a.nc is not None else d['nc']
    n_params = sum(x.numel() for x in model.parameters())
    print(f'[make_random_weights] cfg={a.cfg} width_multiple={d["width_multiple"]} '
          f'nc={nc} params={n_params / 1e6:.2f}M')

    m = deepcopy(model)
    if a.half:
        m = m.half()

    # attempt_load(export.py) 호환: ckpt["ema" if ckpt.get("ema") else "model"] 사용.
    # ema=None => 'model' 사용. .float().fuse().eval() 은 attempt_load 가 수행.
    ckpt = {
        'epoch': -1,
        'best_fitness': None,
        'model': m,
        'ema': None,
        'updates': None,
        'optimizer': None,
        'training_results': None,
        'wandb_id': None,
    }

    if a.out:
        out = a.out
    else:
        tag = f'-w{int(round(d["width_multiple"] * 1000)):03d}' if a.width is not None else ''
        out = str(ROOT / (Path(a.cfg).stem + tag + '-random.pt'))
    torch.save(ckpt, out)
    print(f'[make_random_weights] saved -> {out}')
    print('[make_random_weights] next: export.py 로 ONNX 변환 후 trtexec --fp16 로 latency 측정 '
          '(SIAV2_DESIGN.md Phase 0 공정성 규정 준수).')


if __name__ == '__main__':
    main()
