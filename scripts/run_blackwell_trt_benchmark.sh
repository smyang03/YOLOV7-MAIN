#!/usr/bin/env bash
set -euo pipefail

# Blackwell / CUDA 13 / TensorRT 10.16 benchmark runner.
# Each MODEL entry is name:onnx:engine:input_size.
# Example:
#   MODELS="l640:runs/onnx/l640.onnx:runs/trt/l640.fp16.engine:640" \
#   bash scripts/run_blackwell_trt_benchmark.sh

TRTEXEC="${TRTEXEC:-trtexec}"
OUT="${OUT:-runs/blackwell_trt}"
PRECISION="${PRECISION:-fp16}"
WARMUP="${WARMUP:-200}"
ITERATIONS="${ITERATIONS:-1000}"
DURATION="${DURATION:-30}"
WORKSPACE="${WORKSPACE:-4096}"
MODELS="${MODELS:-}"

if [[ -z "$MODELS" ]]; then
  echo "MODELS is required: name:onnx:engine:input_size[,name:onnx:engine:input_size...]" >&2
  exit 2
fi

mkdir -p "$OUT/logs" "$OUT/engines" "$OUT/meta"

{
  date -Is
  echo "repo=$(pwd)"
  echo "trtexec=$TRTEXEC"
  echo "precision=$PRECISION"
  echo "warmup=$WARMUP iterations=$ITERATIONS duration=$DURATION workspace=$WORKSPACE"
  echo "--- trtexec --version ---"
  "$TRTEXEC" --version
  echo "--- nvidia-smi ---"
  nvidia-smi || true
  echo "--- nvcc --version ---"
  nvcc --version || true
  echo "--- python packages ---"
  python - <<'PY'
try:
    import torch
    print('torch', torch.__version__, 'cuda', torch.version.cuda)
except Exception as e:
    print('torch_error', repr(e))
try:
    import tensorrt as trt
    print('tensorrt', trt.__version__)
except Exception as e:
    print('tensorrt_error', repr(e))
PY
} | tee "$OUT/meta/environment.txt"

run_one() {
  local name="$1"; local onnx="$2"; local engine="$3"; local size="$4"
  local build_log="$OUT/logs/${name}.${PRECISION}.build.log"
  local infer_log="$OUT/logs/${name}.${PRECISION}.inference.log"
  local transfer_log="$OUT/logs/${name}.${PRECISION}.transfer.log"
  mkdir -p "$(dirname "$engine")"

  [[ -f "$onnx" ]] || { echo "[SKIP] $name missing ONNX: $onnx"; return 0; }

  local precision_args=()
  case "$PRECISION" in
    fp16) precision_args+=(--fp16) ;;
    fp32) ;;
    fp8) precision_args+=(--fp8) ;;
    fp4|nvfp4)
      echo "[INFO] FP4 requires an explicit NVFP4/quantized graph; no implicit --fp4 is used." >&2
      echo "[SKIP] $name precision=$PRECISION"
      return 0
      ;;
    *) echo "Unsupported PRECISION=$PRECISION" >&2; return 2 ;;
  esac

  echo "[BUILD] $name size=$size precision=$PRECISION"
  "$TRTEXEC" \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    "${precision_args[@]}" \
    --workspace="$WORKSPACE" \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --skipInference \
    2>&1 | tee "$build_log"

  echo "[BENCH] $name GPU compute only"
  "$TRTEXEC" \
    --loadEngine="$engine" \
    --warmUp="$WARMUP" \
    --iterations="$ITERATIONS" \
    --duration="$DURATION" \
    --noDataTransfers \
    --useCudaGraph \
    --useSpinWait \
    --dumpProfile \
    --profilingVerbosity=detailed \
    2>&1 | tee "$infer_log"

  echo "[BENCH] $name H2D/D2H included"
  "$TRTEXEC" \
    --loadEngine="$engine" \
    --warmUp="$WARMUP" \
    --iterations="$ITERATIONS" \
    --duration="$DURATION" \
    --useSpinWait \
    2>&1 | tee "$transfer_log"
}

IFS=',' read -ra entries <<< "$MODELS"
for entry in "${entries[@]}"; do
  IFS=':' read -r name onnx engine size <<< "$entry"
  [[ -n "${name:-}" && -n "${onnx:-}" && -n "${engine:-}" && -n "${size:-}" ]] || {
    echo "Invalid model entry: $entry" >&2; exit 2;
  }
  run_one "$name" "$onnx" "$engine" "$size"
done

echo "Benchmark logs: $OUT/logs"
echo "Environment: $OUT/meta/environment.txt"
