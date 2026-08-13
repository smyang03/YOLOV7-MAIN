# P3 recovery training

The current `p3lite` model has 32 effective P3 channels after
`width_multiple: 0.25`.  The recovery candidates widen only the P3 branch;
P4/P5, the backbone and the inference output format remain unchanged.

Generate the configs from the repository root:

```powershell
python tools\make_p3_recovery_variants.py
```

For the primary scratch experiment, run the existing P3-lite baseline first,
then the two recovery candidates. No teacher or distillation is used:

```powershell
 .\scripts\run_p3_recovery_scratch.ps1 `
  -Data data\siav2.yaml `
  -Candidates p3lite,p3w150,p3w200 `
  -Epochs 300
```

Distillation is optional and should only be run after the scratch ablation:

```powershell
 .\scripts\run_p3_recovery_distill.ps1 `
  -Data data\siav2.yaml `
  -TeacherWeights runs\siav2_train\w6_nc16_teacher\weights\best.pt `
  -Candidates p3w150,p3w200 `
  -Epochs 100

```

`p3w150` uses 192 pre-scaling channels (48 effective channels) and `p3w200`
uses 256 pre-scaling channels (64 effective channels).  Compare mAP,
AP-small and TensorRT FP16 latency under identical settings.
