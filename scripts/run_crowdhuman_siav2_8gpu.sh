#!/usr/bin/env bash
set -euo pipefail

# CrowdHuman/SIAV2 full 8-GPU training workflow.
# Runs sequentially:
#   1) W6 teacher/baseline
#   2) SIAV2 P4/P5/P6 distill
#   3) SIAV2 P3-lite/P4/P5 distill
#   4) SIAV2 P3-lite/P4/P5/P6 distill
#   5) SIAV2 P4/P5/P6 no-distill ablation

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-data/crowdhuman.yaml}"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-100}"
IMG_SIZE="${IMG_SIZE:-1280}"
TEACHER_BATCH="${TEACHER_BATCH:-64}"
STUDENT_BATCH="${STUDENT_BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-64}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-2}"
MASTER_PORT="${MASTER_PORT:-9527}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"
TEACHER_INIT_WEIGHTS="${TEACHER_INIT_WEIGHTS:-}"
STUDENT_INIT_WEIGHTS="${STUDENT_INIT_WEIGHTS:-}"
SKIP_EDA="${SKIP_EDA:-0}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"
SKIP_STUDENTS="${SKIP_STUDENTS:-0}"
SKIP_NODISTILL="${SKIP_NODISTILL:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NOAUTOANCHOR="${NOAUTOANCHOR:-0}"

echo "ROOT=$ROOT"
echo "DATA=$DATA"
echo "DEVICES=$DEVICES NPROC=$NPROC"
echo "EPOCHS=$EPOCHS IMG_SIZE=$IMG_SIZE"
echo "TEACHER_BATCH=$TEACHER_BATCH STUDENT_BATCH=$STUDENT_BATCH EVAL_BATCH=$EVAL_BATCH"

EXTRA_COMMON_ARGS=()
if [[ "$NOAUTOANCHOR" == "1" ]]; then
  EXTRA_COMMON_ARGS+=(--noautoanchor)
fi

run_ddp() {
  local port="$1"
  shift
  python -m torch.distributed.launch \
    --nproc_per_node "$NPROC" \
    --master_port "$port" \
    train_aux.py "$@"
}

if [[ "$SKIP_EDA" != "1" ]]; then
  python tools/siav2_dataset_eda.py \
    --data "$DATA" \
    --imgsz "$IMG_SIZE" \
    --cfg cfg/training/yolov7-w6-nc16.yaml \
    --anchor-t 4.0 \
    --json-out runs/crowdhuman/eda/crowdhuman_eda.json \
    --md-out runs/crowdhuman/eda/crowdhuman_eda.md \
    --fail-on-invalid
fi

TEACHER_NAME="w6_crowdhuman_teacher"
TEACHER_WEIGHTS="runs/crowdhuman_train/${TEACHER_NAME}/weights/best.pt"

if [[ "$SKIP_TEACHER" != "1" ]]; then
  TEACHER_ARGS=(
    --data "$DATA"
    --cfg cfg/training/yolov7-w6-nc16.yaml
    --hyp data/hyp.scratch.p6.yaml
    --epochs "$EPOCHS"
    --batch-size "$TEACHER_BATCH"
    --img-size "$IMG_SIZE" "$IMG_SIZE"
    --device "$DEVICES"
    --project runs/crowdhuman_train
    --name "$TEACHER_NAME"
    --workers "$WORKERS"
    --dist-backend "$DIST_BACKEND"
    --seed "$SEED"
    --close-mosaic 20
    --grad-clip 10
    --freeze 0
    "${EXTRA_COMMON_ARGS[@]}"
  )
  if [[ -n "$TEACHER_INIT_WEIGHTS" ]]; then
    TEACHER_ARGS+=(--weights "$TEACHER_INIT_WEIGHTS")
  fi
  run_ddp "$MASTER_PORT" "${TEACHER_ARGS[@]}"
fi

if [[ "$SKIP_STUDENTS" != "1" || "$SKIP_EVAL" != "1" ]]; then
  if [[ ! -f "$TEACHER_WEIGHTS" ]]; then
    echo "Missing teacher weights: $TEACHER_WEIGHTS" >&2
    exit 1
  fi
fi

run_student_distill() {
  local port="$1"
  local name="$2"
  local cfg="$3"
  local hyp="$4"
  shift 4
  local distill_args=("$@")

  STUDENT_ARGS=(
    --data "$DATA"
    --cfg "$cfg"
    --hyp "$hyp"
    --epochs "$EPOCHS"
    --batch-size "$STUDENT_BATCH"
    --img-size "$IMG_SIZE" "$IMG_SIZE"
    --device "$DEVICES"
    --project runs/crowdhuman_distill_train
    --name "$name"
    --workers "$WORKERS"
    --dist-backend "$DIST_BACKEND"
    --seed "$SEED"
    --close-mosaic 20
    --grad-clip 10
    --freeze 0
    --distill
    --teacher-weights "$TEACHER_WEIGHTS"
    --distill-weight 0.25
    --distill-obj-weight 1.0
    --distill-cls-weight 1.0
    --distill-box-weight 0.0
    --distill-temp 2.0
    --distill-conf-thres 0.01
    --distill-small-gain 1.25
    --distill-small-px 128
    "${distill_args[@]}"
    "${EXTRA_COMMON_ARGS[@]}"
  )
  if [[ -n "$STUDENT_INIT_WEIGHTS" ]]; then
    STUDENT_ARGS+=(--weights "$STUDENT_INIT_WEIGHTS")
  fi
  run_ddp "$port" "${STUDENT_ARGS[@]}"
}

run_student_nodistill() {
  local port="$1"
  local name="$2"
  local cfg="$3"
  local hyp="$4"

  STUDENT_ARGS=(
    --data "$DATA"
    --cfg "$cfg"
    --hyp "$hyp"
    --epochs "$EPOCHS"
    --batch-size "$STUDENT_BATCH"
    --img-size "$IMG_SIZE" "$IMG_SIZE"
    --device "$DEVICES"
    --project runs/crowdhuman_ablation_train
    --name "$name"
    --workers "$WORKERS"
    --dist-backend "$DIST_BACKEND"
    --seed "$SEED"
    --close-mosaic 20
    --grad-clip 10
    --freeze 0
    "${EXTRA_COMMON_ARGS[@]}"
  )
  if [[ -n "$STUDENT_INIT_WEIGHTS" ]]; then
    STUDENT_ARGS+=(--weights "$STUDENT_INIT_WEIGHTS")
  fi
  run_ddp "$port" "${STUDENT_ARGS[@]}"
}

if [[ "$SKIP_STUDENTS" != "1" ]]; then
  run_student_distill "$((MASTER_PORT + 1))" \
    siav2_p4p6_w250_crowdhuman_distill \
    cfg/training/yolov7-l6-siav2-p4p6-pruned-w250.yaml \
    data/hyp.siav2-p4small-aux-relaxed.yaml \
    --distill-strides 16 32 64 \
    --distill-cross-weight 0.5 \
    --distill-cross-strides 8:16

  run_student_distill "$((MASTER_PORT + 2))" \
    siav2_p3lite_p4p5_w250_crowdhuman_distill \
    cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml \
    data/hyp.siav2-p3lite-aux-relaxed.yaml \
    --distill-strides 8 16 32

  run_student_distill "$((MASTER_PORT + 3))" \
    siav2_p3lite_p4p6_w250_crowdhuman_distill \
    cfg/training/yolov7-l6-siav2-p3lite-p4p6-w250.yaml \
    data/hyp.siav2-p3lite-p4p6-aux-relaxed.yaml \
    --distill-strides 8 16 32 64

  if [[ "$SKIP_NODISTILL" != "1" ]]; then
    run_student_nodistill "$((MASTER_PORT + 101))" \
      siav2_p4p6_w250_crowdhuman_nodistill \
      cfg/training/yolov7-l6-siav2-p4p6-pruned-w250.yaml \
      data/hyp.siav2-p4small-aux-relaxed.yaml
  fi
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  EVAL_DEVICE="${DEVICES%%,*}"
  declare -a EVAL_MODELS=(
    "w6_crowdhuman_teacher:${TEACHER_WEIGHTS}"
    "siav2_p4p6_w250_crowdhuman_distill:runs/crowdhuman_distill_train/siav2_p4p6_w250_crowdhuman_distill/weights/best.pt"
    "siav2_p3lite_p4p5_w250_crowdhuman_distill:runs/crowdhuman_distill_train/siav2_p3lite_p4p5_w250_crowdhuman_distill/weights/best.pt"
    "siav2_p3lite_p4p6_w250_crowdhuman_distill:runs/crowdhuman_distill_train/siav2_p3lite_p4p6_w250_crowdhuman_distill/weights/best.pt"
  )
  if [[ "$SKIP_NODISTILL" != "1" ]]; then
    EVAL_MODELS+=("siav2_p4p6_w250_crowdhuman_nodistill:runs/crowdhuman_ablation_train/siav2_p4p6_w250_crowdhuman_nodistill/weights/best.pt")
  fi

  for item in "${EVAL_MODELS[@]}"; do
    name="${item%%:*}"
    weights="${item#*:}"
    if [[ ! -f "$weights" ]]; then
      echo "Skipping eval for $name, missing $weights" >&2
      continue
    fi
    python tools/eval_siav2_size_ap.py \
      --data "$DATA" \
      --weights "$weights" \
      --img-size "$IMG_SIZE" \
      --batch-size "$EVAL_BATCH" \
      --device "$EVAL_DEVICE" \
      --project runs/crowdhuman_eval \
      --name "$name" \
      --small-max-side 64 \
      --medium-max-side 128 \
      --exist-ok
  done
fi
