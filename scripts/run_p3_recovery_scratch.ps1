param(
    [string]$Data = "data\siav2.yaml",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [int]$BatchSize = 4,
    [int]$ImgSize = 1280,
    [string]$Project = "runs\siav2_p3_recovery_scratch",
    [string[]]$Candidates = @("p3lite", "p3w150", "p3w200"),
    [int]$Seed = 2,
    [int]$Workers = 8,
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$candidateMap = @{
    "p3lite" = "cfg\training\yolov7-l6-siav2-p3lite-p4p5-w250.yaml"
    "p3w150" = "cfg\training\yolov7-l6-siav2-p3lite-p3w150-w250.yaml"
    "p3w200" = "cfg\training\yolov7-l6-siav2-p3lite-p3w200-w250.yaml"
}

foreach ($candidate in $Candidates) {
    if (-not $candidateMap.ContainsKey($candidate)) {
        throw "Unknown candidate '$candidate'. Use: $($candidateMap.Keys -join ', ')"
    }

    $runArgs = @(
        "train_aux.py",
        "--data", $Data,
        "--cfg", $candidateMap[$candidate],
        "--hyp", "data\hyp.siav2-p3lite-aux-relaxed.yaml",
        "--weights", "",
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--img-size", "$ImgSize", "$ImgSize",
        "--device", $Device,
        "--workers", "$Workers",
        "--project", $Project,
        "--name", "${candidate}_scratch",
        "--seed", "$Seed",
        "--close-mosaic", "20",
        "--grad-clip", "10",
        "--freeze", "0"
    )

    if ($NoAutoAnchor) { $runArgs += "--noautoanchor" }
    conda run --no-capture-output -n yolov7 python @runArgs
}
