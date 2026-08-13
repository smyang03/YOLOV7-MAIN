#!/usr/bin/env bash
set -uo pipefail

DATA="${1:-data/siav2.yaml}"
DEVICE="${2:-0}"
BATCH="${3:-32}"
EPOCHS="${4:-100}"
IMG_SIZE="${5:-1280}"
PROJECT="${6:-runs/siav2_full_scratch}"

models=(
  s01_p3lite_p4p5_w025
  s02_p3w150_p4p5_w025
  s03_p3w200_p4p5_w025
  s04_p3lite_p4p6_w025
  s05_p3w150_p4p6_w025
  s06_p3w200_p4p6_w025
  s07_p3w200_p4p5_w031
  s08_p3w200_p4p6_w031
  s09_p3w200_p4p6_w0375
  s10_p3w200_p4p6_w0375_deep
  s11_p3w200_p4p6_w0375_small
  s12_p3w200_p4p6_w050
)

for name in "${models[@]}"; do
  cfg="cfg/training/${name}.yaml"
  hyp="data/hyp.siav2-p3lite-aux-relaxed.yaml"
  if [[ "$name" == *_small ]]; then
    hyp="data/hyp.siav2-p3-recovery-small.yaml"
  elif [[ "$name" == *_p4p6_* ]]; then
    hyp="data/hyp.siav2-p3lite-p4p6-aux-relaxed.yaml"
  fi
  best="${PROJECT}/${name}/weights/best.pt"

  if [[ -f "$best" ]]; then
    echo "[SKIP] $name already has $best"
    continue
  fi

  echo "[START] $name"
  python train_aux.py \
    --data "$DATA" \
    --cfg "$cfg" \
    --hyp "$hyp" \
    --weights '' \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --img-size "$IMG_SIZE" "$IMG_SIZE" \
    --device "$DEVICE" \
    --workers 8 \
    --project "$PROJECT" \
    --name "$name" \
    --seed 2 \
    --close-mosaic 30 \
    --grad-clip 10 \
    --freeze 0

  status=$?
  if [[ $status -ne 0 ]]; then
    echo "[FAILED] $name exit=$status"
    exit "$status"
  fi
  echo "[DONE] $name"
done

echo "All scratch candidates completed."
