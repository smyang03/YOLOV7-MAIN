param(
    [string]$Data = "data\crowdhuman.yaml",
    [string]$Devices = "0,1",
    [int]$NumProc = 2,
    [int]$TeacherBatchSize = 8,
    [int]$StudentBatchSize = 8,
    [int]$EvalBatchSize = 8,
    [int]$ImgSize = 1280,
    [int]$Epochs = 100,
    [int]$Workers = 8,
    [string]$DistBackend = "auto",
    [string]$TeacherInitWeights = "",
    [string]$StudentInitWeights = "",
    [int[]]$Seeds = @(2),
    [string[]]$Candidates = @("p4p6", "p3lite", "p3lite_p4p6"),
    [string[]]$NoDistillCandidates = @("p4p6"),
    [int]$MasterPort = 9527,
    [switch]$SkipEda,
    [switch]$SkipTeacher,
    [switch]$SkipStudents,
    [switch]$SkipNoDistill,
    [switch]$SkipEval,
    [switch]$NoAutoAnchor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

function Invoke-DDPTrain {
    param(
        [int]$Port,
        [string[]]$TrainArgs
    )
    conda run --no-capture-output -n yolov7 python -m torch.distributed.launch `
        --nproc_per_node $NumProc `
        --master_port $Port `
        train_aux.py @TrainArgs
}

function Get-SeedSuffix {
    param([int]$Seed)
    if ($Seeds.Count -gt 1) {
        return "_seed$Seed"
    }
    return ""
}

if (-not $SkipEda) {
    conda run --no-capture-output -n yolov7 python tools\siav2_dataset_eda.py `
        --data $Data `
        --imgsz $ImgSize `
        --cfg cfg\training\yolov7-w6-nc16.yaml `
        --anchor-t 4.0 `
        --json-out runs\crowdhuman\eda\crowdhuman_eda.json `
        --md-out runs\crowdhuman\eda\crowdhuman_eda.md `
        --fail-on-invalid
}

$candidateMap = @{
    "p4p6" = @{
        BaseName = "siav2_p4p6_w250"
        Cfg = "cfg\training\yolov7-l6-siav2-p4p6-pruned-w250.yaml"
        Hyp = "data\hyp.siav2-p4small-aux-relaxed.yaml"
        Strides = @("16", "32", "64")
        Cross = @("8:16")
        CrossWeight = "0.5"
    }
    "p3lite" = @{
        BaseName = "siav2_p3lite_p4p5_w250"
        Cfg = "cfg\training\yolov7-l6-siav2-p3lite-p4p5-w250.yaml"
        Hyp = "data\hyp.siav2-p3lite-aux-relaxed.yaml"
        Strides = @("8", "16", "32")
        Cross = @()
        CrossWeight = "0.0"
    }
    "p3lite_p4p6" = @{
        BaseName = "siav2_p3lite_p4p6_w250"
        Cfg = "cfg\training\yolov7-l6-siav2-p3lite-p4p6-w250.yaml"
        Hyp = "data\hyp.siav2-p3lite-p4p6-aux-relaxed.yaml"
        Strides = @("8", "16", "32", "64")
        Cross = @()
        CrossWeight = "0.0"
    }
}

foreach ($seed in $Seeds) {
    $suffix = Get-SeedSuffix -Seed $seed
    $teacherName = "w6_crowdhuman_teacher$suffix"
    $teacherWeights = "runs\crowdhuman_train\$teacherName\weights\best.pt"

    if (-not $SkipTeacher) {
        $teacherArgs = @(
            "--data", $Data,
            "--cfg", "cfg\training\yolov7-w6-nc16.yaml",
            "--hyp", "data\hyp.scratch.p6.yaml",
            "--epochs", "$Epochs",
            "--batch-size", "$TeacherBatchSize",
            "--img-size", "$ImgSize", "$ImgSize",
            "--device", $Devices,
            "--project", "runs\crowdhuman_train",
            "--name", $teacherName,
            "--workers", "$Workers",
            "--dist-backend", $DistBackend,
            "--seed", "$seed",
            "--close-mosaic", "20",
            "--grad-clip", "10",
            "--freeze", "0"
        )
        if ($TeacherInitWeights -ne "") {
            $teacherArgs += @("--weights", $TeacherInitWeights)
        }
        if ($NoAutoAnchor) {
            $teacherArgs += "--noautoanchor"
        }
        Invoke-DDPTrain -Port ($MasterPort + ($seed % 1000)) -TrainArgs $teacherArgs
    }

    $needsTeacherWeights = (-not $SkipStudents) -or (-not $SkipEval)
    if ($needsTeacherWeights -and -not (Test-Path $teacherWeights)) {
        throw "Teacher weights not found: $teacherWeights"
    }

    if (-not $SkipStudents) {
        $idx = 1
        foreach ($candidate in $Candidates) {
            if (-not $candidateMap.ContainsKey($candidate)) {
                throw "Unknown candidate '$candidate'. Use one of: $($candidateMap.Keys -join ', ')"
            }
            $item = $candidateMap[$candidate]
            $studentName = "$($item.BaseName)_crowdhuman_distill$suffix"
            $studentArgs = @(
                "--data", $Data,
                "--cfg", $item.Cfg,
                "--hyp", $item.Hyp,
                "--epochs", "$Epochs",
                "--batch-size", "$StudentBatchSize",
                "--img-size", "$ImgSize", "$ImgSize",
                "--device", $Devices,
                "--project", "runs\crowdhuman_distill_train",
                "--name", $studentName,
                "--workers", "$Workers",
                "--dist-backend", $DistBackend,
                "--seed", "$seed",
                "--close-mosaic", "20",
                "--grad-clip", "10",
                "--freeze", "0",
                "--distill",
                "--teacher-weights", $teacherWeights,
                "--distill-weight", "0.25",
                "--distill-obj-weight", "1.0",
                "--distill-cls-weight", "1.0",
                "--distill-box-weight", "0.0",
                "--distill-temp", "2.0",
                "--distill-conf-thres", "0.01",
                "--distill-small-gain", "1.25",
                "--distill-small-px", "128",
                "--distill-strides"
            ) + $item.Strides
            if ($item.Cross.Count -gt 0) {
                $studentArgs += @("--distill-cross-weight", $item.CrossWeight, "--distill-cross-strides") + $item.Cross
            }
            if ($StudentInitWeights -ne "") {
                $studentArgs += @("--weights", $StudentInitWeights)
            }
            if ($NoAutoAnchor) {
                $studentArgs += "--noautoanchor"
            }
            Invoke-DDPTrain -Port ($MasterPort + ($seed % 1000) + $idx) -TrainArgs $studentArgs
            $idx += 1
        }
    }

    if (-not $SkipStudents -and -not $SkipNoDistill) {
        $idx = 101
        foreach ($candidate in $NoDistillCandidates) {
            if (-not $candidateMap.ContainsKey($candidate)) {
                throw "Unknown no-distill candidate '$candidate'. Use one of: $($candidateMap.Keys -join ', ')"
            }
            $item = $candidateMap[$candidate]
            $studentName = "$($item.BaseName)_crowdhuman_nodistill$suffix"
            $studentArgs = @(
                "--data", $Data,
                "--cfg", $item.Cfg,
                "--hyp", $item.Hyp,
                "--epochs", "$Epochs",
                "--batch-size", "$StudentBatchSize",
                "--img-size", "$ImgSize", "$ImgSize",
                "--device", $Devices,
                "--project", "runs\crowdhuman_ablation_train",
                "--name", $studentName,
                "--workers", "$Workers",
                "--dist-backend", $DistBackend,
                "--seed", "$seed",
                "--close-mosaic", "20",
                "--grad-clip", "10",
                "--freeze", "0"
            )
            if ($StudentInitWeights -ne "") {
                $studentArgs += @("--weights", $StudentInitWeights)
            }
            if ($NoAutoAnchor) {
                $studentArgs += "--noautoanchor"
            }
            Invoke-DDPTrain -Port ($MasterPort + ($seed % 1000) + $idx) -TrainArgs $studentArgs
            $idx += 1
        }
    }

    if (-not $SkipEval) {
        $evalModels = @(
            @{ Name = $teacherName; Weights = $teacherWeights }
        )
        foreach ($candidate in $Candidates) {
            if (-not $candidateMap.ContainsKey($candidate)) {
                continue
            }
            $item = $candidateMap[$candidate]
            $name = "$($item.BaseName)_crowdhuman_distill$suffix"
            $evalModels += @{ Name = $name; Weights = "runs\crowdhuman_distill_train\$name\weights\best.pt" }
        }
        if (-not $SkipNoDistill) {
            foreach ($candidate in $NoDistillCandidates) {
                if (-not $candidateMap.ContainsKey($candidate)) {
                    continue
                }
                $item = $candidateMap[$candidate]
                $name = "$($item.BaseName)_crowdhuman_nodistill$suffix"
                $evalModels += @{ Name = $name; Weights = "runs\crowdhuman_ablation_train\$name\weights\best.pt" }
            }
        }

        foreach ($model in $evalModels) {
            if (-not (Test-Path $model.Weights)) {
                Write-Warning "Skipping $($model.Name): missing weights $($model.Weights)"
                continue
            }
            conda run --no-capture-output -n yolov7 python tools\eval_siav2_size_ap.py `
                --data $Data `
                --weights $model.Weights `
                --img-size $ImgSize `
                --batch-size $EvalBatchSize `
                --device ($Devices.Split(",")[0]) `
                --project runs\crowdhuman_eval `
                --name $model.Name `
                --small-max-side 64 `
                --medium-max-side 128 `
                --exist-ok
        }
    }
}
