param(
    [string]$Data = "data\siav2.yaml",
    [string]$Device = "0",
    [int]$ImgSize = 1280,
    [int]$TeacherEpochs = 100,
    [int]$StudentEpochs = 100,
    [int]$TeacherBatchSize = 2,
    [int]$StudentBatchSize = 4,
    [int]$EvalBatchSize = 8,
    [string]$TeacherInitWeights = "",
    [int]$Seed = 2,
    [string[]]$Candidates = @("p4p6", "p3lite", "p3lite_p4p6"),
    [int[]]$TeacherFreeze = @(0),
    [int[]]$StudentFreeze = @(0),
    [switch]$SkipEda,
    [switch]$SkipTeacher,
    [switch]$SkipStudents,
    [switch]$SkipEval,
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not $SkipEda) {
    conda run --no-capture-output -n yolov7 python tools\siav2_dataset_eda.py `
        --data $Data `
        --imgsz $ImgSize `
        --cfg cfg\training\yolov7-w6-nc16.yaml `
        --anchor-t 4.0 `
        --json-out runs\siav2\eda\siav2_dataset_eda.json `
        --md-out runs\siav2\eda\siav2_dataset_eda.md `
        --fail-on-invalid
}

if (-not $SkipTeacher) {
    $teacherArgs = @(
        "-Data", $Data,
        "-Device", $Device,
        "-Epochs", "$TeacherEpochs",
        "-BatchSize", "$TeacherBatchSize",
        "-ImgSize", "$ImgSize",
        "-Seed", "$Seed",
        "-Freeze"
    )
    $teacherArgs += @($TeacherFreeze | ForEach-Object { "$_" })
    if ($TeacherInitWeights -ne "") {
        $teacherArgs += @("-Weights", $TeacherInitWeights)
    }
    if ($NoAutoAnchor) {
        $teacherArgs += "-NoAutoAnchor"
    }
    & "$root\scripts\run_siav2_w6_teacher.ps1" @teacherArgs
}

$teacherWeights = "runs\siav2_train\w6_nc16_teacher\weights\best.pt"
if (-not (Test-Path $teacherWeights)) {
    throw "Teacher weights not found: $teacherWeights"
}

if (-not $SkipStudents) {
    $studentArgs = @(
        "-Data", $Data,
        "-TeacherWeights", $teacherWeights,
        "-Device", $Device,
        "-Epochs", "$StudentEpochs",
        "-BatchSize", "$StudentBatchSize",
        "-ImgSize", "$ImgSize",
        "-Seed", "$Seed",
        "-Candidates"
    ) + $Candidates + @("-Freeze") + @($StudentFreeze | ForEach-Object { "$_" })
    if ($NoAutoAnchor) {
        $studentArgs += "-NoAutoAnchor"
    }
    & "$root\scripts\run_siav2_distill_candidates.ps1" @studentArgs
}

if (-not $SkipEval) {
    & "$root\scripts\run_siav2_eval_matrix.ps1" `
        -Data $Data `
        -Device $Device `
        -BatchSize $EvalBatchSize `
        -ImgSize $ImgSize
}
