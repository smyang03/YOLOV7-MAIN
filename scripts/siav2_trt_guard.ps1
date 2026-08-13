function Assert-SIAV2TensorRTVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Trtexec,
        [string[]]$AllowedPrefixes = @("8.6.1", "10.14")
    )

    if (-not (Test-Path $Trtexec)) {
        throw "trtexec not found: $Trtexec"
    }

    $versionOutput = & $Trtexec --version 2>&1
    $versionText = ($versionOutput | Out-String).Trim()
    $version = $null
    if ($versionText -match "TensorRT\s+version:\s*([0-9]+(?:\.[0-9]+)+)") {
        $version = $Matches[1]
    } elseif ($versionText -match "Version:\s*([0-9]+(?:\.[0-9]+)+)") {
        $version = $Matches[1]
    } elseif ($versionText -match "([0-9]+(?:\.[0-9]+)+)") {
        $version = $Matches[1]
    }

    if (-not $version) {
        throw "Unable to parse TensorRT version from: $versionText"
    }

    $allowed = $false
    foreach ($prefix in $AllowedPrefixes) {
        if ($version.StartsWith($prefix)) {
            $allowed = $true
            break
        }
    }

    if (-not $allowed) {
        throw "Unsupported TensorRT trtexec version '$version'. SIAV2 latency is locked to TensorRT 8.6.1.x or 10.14.x only."
    }

    Write-Host "SIAV2 TensorRT latency version: $version"
    return $version
}
