# SIAV2 Training-Ready Workflow

Date: 2026-07-13

This repo now contains the pieces needed to train and compare the SIAV2 candidates once `data/siav2.yaml` and the dataset are present.

## What Is Ready

- W6 nc16 teacher/baseline training entrypoint.
- SIAV2 P4/P5/P6 w250 distillation candidate.
- SIAV2 P3-lite/P4/P5 w250 distillation candidate.
- SIAV2 P3-lite/P4/P5/P6 w250 distillation candidate.
- P4/P5/P6 P3 compensation via optional cross-stride response distillation `8:16`.
- Dataset EDA with label validity checks.
- Size-bucket AP evaluation for all/small/medium/large targets.
- Full PowerShell workflow wrapper.
- CrowdHuman 100-epoch DDP workflow wrapper.
- Distillation loss is scaled to the same batch-size convention as the YOLO detection loss.

## Required Inputs

- `data/siav2.yaml`
- SIAV2 YOLO labels with class ids `0..15`
- Optional W6 initialization weights, if available
- For CrowdHuman, pass the existing dataset YAML with `-Data`; this repo intentionally does not create or modify the dataset YAML.

## 1. Dataset Gate

```powershell
conda run --no-capture-output -n yolov7 python tools\siav2_dataset_eda.py `
  --data data\siav2.yaml `
  --imgsz 1280 `
  --cfg cfg\training\yolov7-w6-nc16.yaml `
  --anchor-t 4.0 `
  --json-out runs\siav2\eda\siav2_dataset_eda.json `
  --md-out runs\siav2\eda\siav2_dataset_eda.md `
  --fail-on-invalid
```

Training should not start if `invalid_label_lines` is nonzero. Missing label files are reported separately because they may represent valid negative images.

## 2. Train W6 Teacher

```powershell
scripts\run_siav2_w6_teacher.ps1 `
  -Data data\siav2.yaml `
  -Device 0 `
  -Epochs 100 `
  -BatchSize 2 `
  -ImgSize 1280 `
  -Freeze 0
```

Output teacher path:

```text
runs\siav2_train\w6_nc16_teacher\weights\best.pt
```

## 3. Train SIAV2 Candidates

```powershell
scripts\run_siav2_distill_candidates.ps1 `
  -Data data\siav2.yaml `
  -TeacherWeights runs\siav2_train\w6_nc16_teacher\weights\best.pt `
  -Device 0 `
  -Epochs 100 `
  -BatchSize 4 `
  -ImgSize 1280 `
  -Candidates p4p6,p3lite,p3lite_p4p6 `
  -Freeze 0
```

Candidate behavior:

| Candidate | CFG | Hyp | Distillation |
|---|---|---|---|
| P4/P5/P6 w250 | `cfg/training/yolov7-l6-siav2-p4p6-pruned-w250.yaml` | `data/hyp.siav2-p4small-aux-relaxed.yaml` | same-stride `16 32 64` + cross `8:16` |
| P3-lite/P4/P5 w250 | `cfg/training/yolov7-l6-siav2-p3lite-p4p5-w250.yaml` | `data/hyp.siav2-p3lite-aux-relaxed.yaml` | same-stride `8 16 32` |
| P3-lite/P4/P5/P6 w250 | `cfg/training/yolov7-l6-siav2-p3lite-p4p6-w250.yaml` | `data/hyp.siav2-p3lite-p4p6-aux-relaxed.yaml` | same-stride `8 16 32 64` |

## 4. Evaluate Accuracy By Size

```powershell
scripts\run_siav2_eval_matrix.ps1 `
  -Data data\siav2.yaml `
  -Device 0 `
  -BatchSize 8 `
  -ImgSize 1280
```

This writes one `size_ap.json` and `size_ap.md` per model under `runs\siav2_eval`.

Size buckets use max box side in input image pixels after dataloader letterbox. At `1280`, the default buckets are:

- small: `<=64 px`
- medium: `>64 px` and `<=128 px`
- large: `>128 px`

Decision table to fill:

| Model | mAP50 | mAP50-95 | small AP | medium AP | large AP | TRT FP16 1280 avg | TRT FP16 1280 median |
|---|---:|---:|---:|---:|---:|---:|---:|
| W6 nc16 teacher | | | | | | | |
| SIAV2 P4/P5/P6 distill | | | | | | | |
| SIAV2 P3-lite/P4/P5 distill | | | | | | | |
| SIAV2 P3-lite/P4/P5/P6 distill | | | | | | | |

## 5. One-Command Workflow

```powershell
scripts\run_siav2_training_workflow.ps1 `
  -Data data\siav2.yaml `
  -Device 0 `
  -TeacherEpochs 100 `
  -StudentEpochs 100 `
  -TeacherBatchSize 2 `
  -StudentBatchSize 4 `
  -EvalBatchSize 8 `
  -TeacherFreeze 0 `
  -StudentFreeze 0
```

Useful switches:

- `-SkipTeacher` if W6 teacher already exists.
- `-SkipStudents` to only run EDA/teacher/eval.
- `-SkipEval` to stop after training.
- `-NoAutoAnchor` only when anchors are intentionally frozen.
- `-TeacherFreeze` and `-StudentFreeze` follow YOLOv7 layer-freeze syntax, e.g. `50` freezes `model.0` through `model.49`.

## 6. CrowdHuman DDP 100-Epoch Workflow

Use this when the CrowdHuman dataset YAML already exists locally. The script runs EDA, trains the W6 nc16 teacher, distills all three SIAV2 candidates for 100 epochs, and then evaluates size-bucket AP.

```powershell
scripts\run_crowdhuman_siav2_ddp_100.ps1 `
  -Data data\crowdhuman.yaml `
  -Devices 0,1 `
  -NumProc 2 `
  -TeacherBatchSize 8 `
  -StudentBatchSize 8 `
  -EvalBatchSize 8 `
  -Epochs 100 `
  -ImgSize 1280 `
  -Seeds 2
```

If the YAML is elsewhere, only replace `-Data`:

```powershell
scripts\run_crowdhuman_siav2_ddp_100.ps1 `
  -Data E:\datasets\CrowdHuman\crowdhuman.yaml `
  -Devices 0,1 `
  -NumProc 2 `
  -Epochs 100
```

The wrapper passes `--dist-backend auto` to `train_aux.py`. Auto uses NCCL when available and falls back to Gloo otherwise. For native Windows DDP, add `-DistBackend gloo`; for Linux/WSL2 CUDA, leave the default. For four GPUs, use `-Devices 0,1,2,3 -NumProc 4` and raise batch sizes only if memory allows.

The CrowdHuman wrapper also runs a no-distill ablation for `p4p6` by default. For a stricter 3-seed check:

```powershell
scripts\run_crowdhuman_siav2_ddp_100.ps1 `
  -Data data\crowdhuman.yaml `
  -Devices 0,1 `
  -NumProc 2 `
  -Epochs 100 `
  -Seeds 2,12,22
```

Use `-TeacherInitWeights` for a pretrained W6 baseline and `-StudentInitWeights` for a compatible student initialization checkpoint. If those are omitted, `train_aux.py` now defaults to an empty `--weights` value, so it does not accidentally load the original `yolo7.pt` default.

## 7. TensorRT Latency Version Policy

Official SIAV2 latency comparisons must use only:

- TensorRT `8.6.1.x`
- TensorRT `10.14.x`

The SIAV2 TRT profiling wrappers call `scripts\siav2_trt_guard.ps1` and stop if `trtexec --version` is outside those prefixes. Python TensorRT bindings from other versions, such as `10.7`, are not accepted as latency evidence. Pass a specific binary when needed:

```powershell
scripts\profile_siav2_p3_tradeoff_trt.ps1 `
  -Trtexec "C:\TensorRT-10.14.1\bin\trtexec.exe"

scripts\profile_siav2_p3_tradeoff_trt.ps1 `
  -Trtexec "C:\TensorRT-8.6.1\bin\trtexec.exe"
```

## Current Guardrails

- P4/P5/P6 remains the fastest candidate, but it still removes the real P3 head.
- Cross-stride `8:16` distillation is a training-time compensation, not a full replacement for stride-8 inference.
- If small AP drops too much, select P3-lite or revisit width/INT8 tradeoff.
- Feature distillation is still not implemented; add it only if response/cross-response distillation is insufficient.
- Latency tables are not final unless produced by TensorRT `8.6.1.x` or `10.14.x`.
