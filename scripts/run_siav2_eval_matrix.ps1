param(
    [string]$Data = "data\siav2.yaml",
    [string]$Device = "0",
    [int]$BatchSize = 8,
    [int]$ImgSize = 1280,
    [string]$Project = "runs\siav2_eval",
    [string]$W6Weights = "runs\siav2_train\w6_nc16_teacher\weights\best.pt",
    [string]$P4P6Weights = "runs\siav2_distill_train\siav2_p4p6_w250_distill\weights\best.pt",
    [string]$P3LiteWeights = "runs\siav2_distill_train\siav2_p3lite_p4p5_w250_distill\weights\best.pt",
    [string]$P3LiteP4P6Weights = "runs\siav2_distill_train\siav2_p3lite_p4p6_w250_distill\weights\best.pt",
    [double]$SmallMaxSide = 64,
    [double]$MediumMaxSide = 128
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$models = @(
    @{ Name = "w6_nc16_teacher"; Weights = $W6Weights },
    @{ Name = "siav2_p4p6_w250_distill"; Weights = $P4P6Weights },
    @{ Name = "siav2_p3lite_p4p5_w250_distill"; Weights = $P3LiteWeights },
    @{ Name = "siav2_p3lite_p4p6_w250_distill"; Weights = $P3LiteP4P6Weights }
)

foreach ($model in $models) {
    if (-not (Test-Path $model.Weights)) {
        Write-Warning "Skipping $($model.Name): missing weights $($model.Weights)"
        continue
    }
    conda run --no-capture-output -n yolov7 python tools\eval_siav2_size_ap.py `
        --data $Data `
        --weights $model.Weights `
        --img-size $ImgSize `
        --batch-size $BatchSize `
        --device $Device `
        --project $Project `
        --name $model.Name `
        --small-max-side $SmallMaxSide `
        --medium-max-side $MediumMaxSide `
        --exist-ok
}
