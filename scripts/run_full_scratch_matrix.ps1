param(
    [string]$Data = "data\siav2.yaml",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [int]$BatchSize = 4,
    [int]$ImgSize = 1280,
    [string]$Project = "runs\siav2_full_scratch",
    [string[]]$Candidates = @(
        "s01_p3lite_p4p5_w025", "s02_p3w150_p4p5_w025", "s03_p3w200_p4p5_w025",
        "s04_p3lite_p4p6_w025", "s05_p3w150_p4p6_w025", "s06_p3w200_p4p6_w025",
        "s07_p3w200_p4p5_w031", "s08_p3w200_p4p6_w031", "s09_p3w200_p4p6_w0375",
        "s10_p3w200_p4p6_w0375_deep", "s11_p3w200_p4p6_w0375_small", "s12_p3w200_p4p6_w050"
    ),
    [int]$Seed = 2,
    [int]$Workers = 8,
    [switch]$SkipExisting,
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

foreach ($candidate in $Candidates) {
    $cfg = "cfg\training\$candidate.yaml"
    if (-not (Test-Path $cfg)) { throw "Missing config: $cfg. Run tools\make_full_scratch_matrix.py first." }
    $hyp = "data\hyp.siav2-p3lite-aux-relaxed.yaml"
    if ($candidate -like "*_small") {
        $hyp = "data\hyp.siav2-p3-recovery-small.yaml"
    } elseif ($candidate -like "*_p4p6_*") {
        $hyp = "data\hyp.siav2-p3lite-p4p6-aux-relaxed.yaml"
    }
    $best = Join-Path (Join-Path $Project $candidate) "weights\best.pt"
    if ($SkipExisting -and (Test-Path $best)) { Write-Host "Skipping completed: $candidate"; continue }
    $runArgs = @(
        "train_aux.py", "--data", $Data, "--cfg", $cfg, "--hyp", $hyp,
        "--weights", "", "--epochs", "$Epochs", "--batch-size", "$BatchSize",
        "--img-size", "$ImgSize", "$ImgSize", "--device", $Device,
        "--workers", "$Workers", "--project", $Project, "--name", $candidate,
        "--seed", "$Seed", "--close-mosaic", "30", "--grad-clip", "10", "--freeze", "0"
    )
    if ($NoAutoAnchor) { $runArgs += "--noautoanchor" }
    Write-Host "Starting scratch candidate: $candidate"
    conda run --no-capture-output -n yolov7 python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $candidate with exit code $LASTEXITCODE" }
}
