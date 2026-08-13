param(
    [string]$Data = "data\siav2.yaml",
    [string]$Weights = "",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [int]$BatchSize = 2,
    [int]$ImgSize = 1280,
    [string]$Project = "runs\siav2_train",
    [string]$Name = "w6_nc16_teacher",
    [string]$Cfg = "cfg\training\yolov7-w6-nc16.yaml",
    [string]$Hyp = "data\hyp.scratch.p6.yaml",
    [int]$Workers = 8,
    [int]$Seed = 2,
    [int[]]$Freeze = @(0),
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$args = @(
    "train_aux.py",
    "--data", $Data,
    "--cfg", $Cfg,
    "--hyp", $Hyp,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--img-size", "$ImgSize", "$ImgSize",
    "--device", $Device,
    "--project", $Project,
    "--name", $Name,
    "--workers", "$Workers",
    "--seed", "$Seed",
    "--close-mosaic", "20",
    "--grad-clip", "10",
    "--freeze"
)

$args += @($Freeze | ForEach-Object { "$_" })

if ($Weights -ne "") {
    $args += @("--weights", $Weights)
}
if ($NoAutoAnchor) {
    $args += "--noautoanchor"
}

conda run --no-capture-output -n yolov7 python @args
