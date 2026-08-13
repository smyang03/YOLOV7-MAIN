param(
    [string]$Data = "data\siav2.yaml",
    [Parameter(Mandatory = $true)][string]$TeacherWeights,
    [string]$Device = "0",
    [int]$Epochs = 100,
    [int]$BatchSize = 4,
    [int]$ImgSize = 1280,
    [string]$Project = "runs\siav2_p3_recovery",
    [string[]]$Candidates = @("p3w150", "p3w200"),
    [string]$Weights = "",
    [double]$DistillWeight = 0.25,
    [double]$DistillSmallGain = 1.25,
    [int]$Seed = 2,
    [int]$Workers = 8,
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$candidateMap = @{
    "p3w150" = @{ Name = "p3lite_w150_distill"; Cfg = "cfg\training\yolov7-l6-siav2-p3lite-p3w150-w250.yaml" }
    "p3w200" = @{ Name = "p3lite_w200_distill"; Cfg = "cfg\training\yolov7-l6-siav2-p3lite-p3w200-w250.yaml" }
}

foreach ($candidate in $Candidates) {
    if (-not $candidateMap.ContainsKey($candidate)) {
        throw "Unknown candidate '$candidate'. Use: $($candidateMap.Keys -join ', ')"
    }

    $item = $candidateMap[$candidate]
    $runArgs = @(
        "train_aux.py",
        "--data", $Data,
        "--cfg", $item.Cfg,
        "--hyp", "data\hyp.siav2-p3lite-aux-relaxed.yaml",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--img-size", "$ImgSize", "$ImgSize",
        "--device", $Device,
        "--workers", "$Workers",
        "--project", $Project,
        "--name", $item.Name,
        "--seed", "$Seed",
        "--close-mosaic", "20",
        "--grad-clip", "10",
        "--freeze", "0",
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
        "--distill-strides", "8", "16", "32"
    )

    if ($Weights -ne "") { $runArgs += @("--weights", $Weights) }
    if ($NoAutoAnchor) { $runArgs += "--noautoanchor" }
    conda run --no-capture-output -n yolov7 python @runArgs
}
