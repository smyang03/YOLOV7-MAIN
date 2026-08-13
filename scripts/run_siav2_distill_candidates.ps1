param(
    [string]$Data = "data\siav2.yaml",
    [string]$TeacherWeights = "runs\siav2_train\w6_nc16_teacher\weights\best.pt",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [int]$BatchSize = 4,
    [int]$ImgSize = 1280,
    [string]$Project = "runs\siav2_distill_train",
    [string[]]$Candidates = @("p4p6", "p3lite"),
    [string]$P4P6Weights = "",
    [string]$P3LiteWeights = "",
    [string]$P3LiteP4P6Weights = "",
    [double]$DistillWeight = 0.25,
    [double]$DistillSmallGain = 1.25,
    [double]$P4P6CrossWeight = 0.5,
    [int]$Seed = 2,
    [int[]]$Freeze = @(0),
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$candidateMap = @{
    "p4p6" = @{
        Name = "siav2_p4p6_w250_distill"
        Cfg = "cfg\training\yolov7-l6-siav2-p4p6-pruned-w250.yaml"
        Hyp = "data\hyp.siav2-p4small-aux-relaxed.yaml"
        Weights = $P4P6Weights
        Strides = @("16", "32", "64")
        CrossStrides = @("8:16")
        CrossWeight = "$P4P6CrossWeight"
    }
    "p3lite" = @{
        Name = "siav2_p3lite_p4p5_w250_distill"
        Cfg = "cfg\training\yolov7-l6-siav2-p3lite-p4p5-w250.yaml"
        Hyp = "data\hyp.siav2-p3lite-aux-relaxed.yaml"
        Weights = $P3LiteWeights
        Strides = @("8", "16", "32")
        CrossStrides = @()
        CrossWeight = "0.0"
    }
    "p3lite_p4p6" = @{
        Name = "siav2_p3lite_p4p6_w250_distill"
        Cfg = "cfg\training\yolov7-l6-siav2-p3lite-p4p6-w250.yaml"
        Hyp = "data\hyp.siav2-p3lite-p4p6-aux-relaxed.yaml"
        Weights = $P3LiteP4P6Weights
        Strides = @("8", "16", "32", "64")
        CrossStrides = @()
        CrossWeight = "0.0"
    }
}

foreach ($candidate in $Candidates) {
    if (-not $candidateMap.ContainsKey($candidate)) {
        throw "Unknown candidate '$candidate'. Use one of: $($candidateMap.Keys -join ', ')"
    }
    $item = $candidateMap[$candidate]
    $args = @(
        "train_aux.py",
        "--data", $Data,
        "--cfg", $item.Cfg,
        "--hyp", $item.Hyp,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--img-size", "$ImgSize", "$ImgSize",
        "--device", $Device,
        "--project", $Project,
        "--name", $item.Name,
        "--seed", "$Seed",
        "--close-mosaic", "20",
        "--grad-clip", "10",
        "--distill",
        "--teacher-weights", $TeacherWeights,
        "--distill-weight", "$DistillWeight",
        "--distill-obj-weight", "1.0",
        "--distill-cls-weight", "1.0",
        "--distill-box-weight", "0.0",
        "--distill-temp", "2.0",
        "--distill-conf-thres", "0.01",
        "--distill-small-gain", "$DistillSmallGain",
        "--distill-small-px", "128",
        "--freeze"
    ) + @($Freeze | ForEach-Object { "$_" }) + @(
        "--distill-strides"
    ) + $item.Strides

    if ($item.CrossStrides.Count -gt 0) {
        $args += @("--distill-cross-weight", $item.CrossWeight, "--distill-cross-strides") + $item.CrossStrides
    }
    if ($item.Weights -ne "") {
        $args += @("--weights", $item.Weights)
    }
    if ($NoAutoAnchor) {
        $args += "--noautoanchor"
    }

    conda run --no-capture-output -n yolov7 python @args
}
