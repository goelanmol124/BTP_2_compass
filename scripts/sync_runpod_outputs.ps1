<#
.SYNOPSIS
  Pull all RunPod ``outputs/`` (trajectories, logs, analysis, etc.) to this repo.

.DESCRIPTION
  The pod should already have ``/tmp/btp_outputs_bundle.tar`` split into
  ``/tmp/btp_parts/p_*`` (400 MiB pieces). This script downloads each piece with
  SSH keepalives, retries on failure, joins them into one ``.tar``, and extracts
  into ``<RepoRoot>\outputs`` (merging / overwriting same paths).

  Tuning for speed vs stability:
  - Default ``-Parallel 1`` (safest on flaky links).
  - Try ``-Parallel 2`` if your connection is stable and you want more throughput.

  Prerequisites: OpenSSH ``scp`` (Windows 10+), SSH key, pod reachable.

.EXAMPLE
  .\scripts\sync_runpod_outputs.ps1 -SshHost 203.57.40.227 -Port 10148 `
    -KeyFile "$env:USERPROFILE\.ssh\id_ed25519" -RepoRoot "C:\Users\you\Projects\BTP\BTP_2_Calm"
#>
[CmdletBinding()]
param(
    [string]$SshHost = "203.57.40.227",
    [int]$Port = 10148,
    [string]$User = "root",
    [string]$KeyFile = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RemoteTarDir = "/tmp/btp_parts",
    [string]$RemoteTarPrefix = "p_",
    [int]$Parallel = 1,
    [int]$MaxRetriesPerPart = 5,
    [int]$RetryDelaySec = 30
)

$ErrorActionPreference = "Stop"
$parts = @("aa", "ab", "ac", "ad", "ae")
$remoteNames = $parts | ForEach-Object { "${RemoteTarPrefix}$_" }
$chunkDir = Join-Path $RepoRoot "btp_parts"
$bundleTar = Join-Path $RepoRoot "btp_outputs_bundle.tar"

$sshBase = @(
    "-i", $KeyFile,
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=720",
    "-o", "TCPKeepAlive=yes",
    "-p", "$Port",
    "${User}@${SshHost}"
)

$scpBase = @(
    "-i", $KeyFile,
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=720",
    "-o", "TCPKeepAlive=yes",
    "-P", "$Port"
)

function Get-RemoteSize {
    param([string]$RemotePath)
    $raw = & ssh @sshBase "stat -c%s $RemotePath 2>/dev/null"
    if ($LASTEXITCODE -ne 0 -or -not $raw) { return $null }
    return [int64]$raw.Trim()
}

function Copy-OnePart {
    param(
        [string]$RemoteName,
        [string]$LocalPath,
        [int64]$ExpectedSize
    )
    for ($attempt = 1; $attempt -le $MaxRetriesPerPart; $attempt++) {
        $remote = "${User}@${SshHost}:${RemoteTarDir}/${RemoteName}"
        Write-Host "[scp] $RemoteName (attempt $attempt / $MaxRetriesPerPart) -> $LocalPath"
        & scp.exe @scpBase $remote $LocalPath
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "scp failed for $RemoteName (exit $LASTEXITCODE)"
            Start-Sleep -Seconds $RetryDelaySec
            continue
        }
        if (-not (Test-Path $LocalPath)) {
            Write-Warning "Local file missing after scp: $LocalPath"
            Start-Sleep -Seconds $RetryDelaySec
            continue
        }
        $len = (Get-Item $LocalPath).Length
        if ($len -ne $ExpectedSize) {
            Write-Warning "Size mismatch for $RemoteName : local $len vs remote $ExpectedSize — retrying"
            Remove-Item -Force $LocalPath -ErrorAction SilentlyContinue
            Start-Sleep -Seconds $RetryDelaySec
            continue
        }
        return $true
    }
    return $false
}

New-Item -ItemType Directory -Force -Path $chunkDir | Out-Null

Write-Host "=== Remote chunk sizes ==="
$sizes = @{}
foreach ($name in $remoteNames) {
    $sz = Get-RemoteSize "/tmp/btp_parts/$name"
    if ($null -eq $sz) { throw "Cannot stat remote /tmp/btp_parts/$name — rebuild chunks on the pod (see script header)." }
    $sizes[$name] = $sz
    Write-Host ("  {0,-6} {1,12} bytes" -f $name, $sz)
}

if ($Parallel -le 1) {
    foreach ($name in $remoteNames) {
        $local = Join-Path $chunkDir $name
        $exp = $sizes[$name]
        if ((Test-Path $local) -and ((Get-Item $local).Length -eq $exp)) {
            Write-Host "[skip] $name already complete ($exp bytes)"
            continue
        }
        if (Test-Path $local) { Remove-Item -Force $local }
        if (-not (Copy-OnePart -RemoteName $name -LocalPath $local -ExpectedSize $exp)) {
            throw "Giving up on $name after $MaxRetriesPerPart attempts."
        }
    }
}
else {
    $jobs = @()
    foreach ($name in $remoteNames) {
        $local = Join-Path $chunkDir $name
        $exp = $sizes[$name]
        if ((Test-Path $local) -and ((Get-Item $local).Length -eq $exp)) {
            Write-Host "[skip] $name already complete"
            continue
        }
        if (Test-Path $local) { Remove-Item -Force $local }
        while (@($jobs | Where-Object { $_.State -eq "Running" }).Count -ge $Parallel) {
            Start-Sleep -Seconds 2
        }
        $n = $name
        $lc = $local
        $ex = $exp
        $jobs += Start-Job -Name "dl_$n" -ScriptBlock {
            param($ScpBase, $User, $SshHost, $Port, $RemoteTarDir, $RemoteName, $LocalPath, $ExpectedSize, $MaxRetriesPerPart, $RetryDelaySec)
            for ($attempt = 1; $attempt -le $MaxRetriesPerPart; $attempt++) {
                $remote = "${User}@${SshHost}:${RemoteTarDir}/${RemoteName}"
                & scp.exe @ScpBase $remote $LocalPath
                if ($LASTEXITCODE -ne 0) { Start-Sleep -Seconds $RetryDelaySec; continue }
                if (-not (Test-Path $LocalPath)) { Start-Sleep -Seconds $RetryDelaySec; continue }
                $len = (Get-Item $LocalPath).Length
                if ($len -ne $ExpectedSize) {
                    Remove-Item -Force $LocalPath -ErrorAction SilentlyContinue
                    Start-Sleep -Seconds $RetryDelaySec
                    continue
                }
                return $true
            }
            return $false
        } -ArgumentList $scpBase, $User, $SshHost, $Port, $RemoteTarDir, $n, $lc, $ex, $MaxRetriesPerPart, $RetryDelaySec
    }
    if ($jobs.Count -gt 0) {
        Wait-Job $jobs | Out-Null
        foreach ($j in $jobs) {
            $ok = Receive-Job $j
            if (-not $ok) { throw "Parallel download failed for job $($j.Name)" }
        }
        Remove-Job $jobs
    }
}

Write-Host "=== Joining chunks -> $bundleTar ==="
if (Test-Path $bundleTar) { Remove-Item -Force $bundleTar }
$joined = ($remoteNames | ForEach-Object { Join-Path $chunkDir $_ }) -join "+"
$cmd = "cmd /c copy /b `"$joined`" `"$bundleTar`""
cmd /c $cmd | Out-Null
if (-not (Test-Path $bundleTar)) { throw "Join failed: $bundleTar missing" }
$sum = ($remoteNames | ForEach-Object { $sizes[$_] } | Measure-Object -Sum).Sum
if ((Get-Item $bundleTar).Length -ne $sum) {
    throw "Joined tarball size mismatch (expected $sum bytes)."
}

Write-Host "=== Extracting tar into $RepoRoot (outputs/) ==="
Push-Location $RepoRoot
try {
    & tar.exe -xf $bundleTar
    if ($LASTEXITCODE -ne 0) { throw "tar extract failed with exit $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "Done. Bundle: $bundleTar (you may delete it and btp_parts\ to save disk)."
