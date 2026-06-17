#!/usr/bin/env pwsh
#
# Windows-native mirror of integration_tests.sh: asserts the `timbal create`
# exit-code / output contract. Runs under PowerShell Core (pwsh) on any OS, so
# it can be developed on macOS/Linux and run for real on windows-latest in CI.
#
#   exit 0  success
#   exit 2  usage / precondition error
#   exit 1  runtime failure
#
#   - errors go to stderr; stdout stays empty on failure
#   - `-q` success prints exactly one line on stdout: the absolute project path
#
# Network-dependent success cases (fetch blueprints from GitHub) run only when
# the env var TIMBAL_CLI_E2E_NETWORK=1.

$ErrorActionPreference = 'Continue'

$ScriptDir = $PSScriptRoot
$Version = if ($env:TIMBAL_CLI_VERSION) { $env:TIMBAL_CLI_VERSION } else { 'dev' }

$script:Pass = 0
$script:Fail = 0

# Empty file used as stdin for every invocation so std-in is never a TTY
# (mirrors `</dev/null`), which deterministically exercises the no-TTY guard.
$EmptyIn = New-TemporaryFile

function Get-Bin {
    if ($IsWindows) { $zos = 'windows'; $ext = '.exe'; $abi = '-gnu' }
    elseif ($IsMacOS) { $zos = 'macos'; $ext = ''; $abi = '' }
    elseif ($IsLinux) { $zos = 'linux'; $ext = ''; $abi = '-gnu' }
    else { Write-Host "Unsupported OS"; exit 1 }

    switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
        'X64' { $zarch = 'x86_64' }
        'Arm64' { $zarch = 'aarch64' }
        default { Write-Host "Unsupported arch"; exit 1 }
    }
    return Join-Path $ScriptDir "zig-out/$Version/timbal-$Version-$zos-$zarch$abi$ext"
}

# Runs the binary, returns @{ Code; Out; Err } with streams captured to files.
function Invoke-Cli {
    param([string]$Bin, [string[]]$CliArgs)
    $outFile = New-TemporaryFile
    $errFile = New-TemporaryFile
    $p = Start-Process -FilePath $Bin -ArgumentList $CliArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardInput $EmptyIn -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $out = (Get-Content -Raw -ErrorAction SilentlyContinue $outFile)
    $err = (Get-Content -Raw -ErrorAction SilentlyContinue $errFile)
    Remove-Item -Force $outFile, $errFile -ErrorAction SilentlyContinue
    return @{ Code = $p.ExitCode; Out = ($out ?? ''); Err = ($err ?? '') }
}

function Assert-Cli {
    param([string]$Desc, [int]$Expected, [string]$Substr, [string]$Bin, [string[]]$CliArgs)
    $r = Invoke-Cli -Bin $Bin -CliArgs $CliArgs
    $ok = $true
    if ($r.Code -ne $Expected) { $ok = $false; Write-Host "  exit: got $($r.Code), want $Expected" -ForegroundColor Red }
    if ($r.Out.Trim().Length -ne 0) { $ok = $false; Write-Host "  stdout not empty: '$($r.Out)'" -ForegroundColor Red }
    if ($Substr -and ($r.Err -notlike "*$Substr*")) {
        $ok = $false; Write-Host "  stderr missing: '$Substr' (got: '$($r.Err.Split("`n")[0])')" -ForegroundColor Red
    }
    if ($ok) { $script:Pass++; Write-Host "ok   $Desc (exit $($r.Code))" -ForegroundColor Green }
    else { $script:Fail++; Write-Host "FAIL $Desc" -ForegroundColor Red }
}

function Assert-SuccessQuiet {
    param([string]$Bin, [string]$Dir)
    $r = Invoke-Cli -Bin $Bin -CliArgs @('create', $Dir, '--agent', 'assistant', '-q')
    $lines = ($r.Out -split "`r?`n" | Where-Object { $_.Trim().Length -gt 0 }).Count
    $ok = $true
    if ($r.Code -ne 0) { $ok = $false; Write-Host "  exit: got $($r.Code), want 0" -ForegroundColor Red }
    if ($lines -ne 1) { $ok = $false; Write-Host "  stdout should be 1 line, got $lines" -ForegroundColor Red }
    if (-not (Test-Path "$Dir/.git")) { $ok = $false; Write-Host "  missing .git/" -ForegroundColor Red }
    if (-not (Test-Path "$Dir/workforce/assistant/timbal.yaml")) { $ok = $false; Write-Host "  missing workforce/assistant/timbal.yaml" -ForegroundColor Red }
    if (-not (Test-Path "$Dir/api")) { $ok = $false; Write-Host "  missing api/" -ForegroundColor Red }
    if (Test-Path "$Dir/ui") { $ok = $false; Write-Host "  unexpected ui/ without --with-ui" -ForegroundColor Red }
    if ($ok) { $script:Pass++; Write-Host "ok   success -q + invariants" -ForegroundColor Green }
    else { $script:Fail++; Write-Host "FAIL success -q + invariants" -ForegroundColor Red }
}

function Assert-SuccessWithUi {
    param([string]$Bin, [string]$Dir)
    $r = Invoke-Cli -Bin $Bin -CliArgs @('create', $Dir, '--agent', 'assistant', '--with-ui', '-q')
    $ok = $true
    if ($r.Code -ne 0) { $ok = $false; Write-Host "  exit: got $($r.Code), want 0" -ForegroundColor Red }
    if (-not (Test-Path "$Dir/ui")) { $ok = $false; Write-Host "  missing ui/ with --with-ui" -ForegroundColor Red }
    if (-not (Test-Path "$Dir/.git")) { $ok = $false; Write-Host "  missing .git/" -ForegroundColor Red }
    if ($ok) { $script:Pass++; Write-Host "ok   success --with-ui" -ForegroundColor Green }
    else { $script:Fail++; Write-Host "FAIL success --with-ui" -ForegroundColor Red }
}

$Bin = Get-Bin
if (-not (Test-Path $Bin)) {
    Write-Host "Building CLI (zig build)..."
    Push-Location $ScriptDir
    zig build
    Pop-Location
}
if (-not (Test-Path $Bin)) { Write-Host "binary not found: $Bin" -ForegroundColor Red; exit 1 }
Write-Host "Using binary: $Bin"

$T = Join-Path ([System.IO.Path]::GetTempPath()) ("timbal-cli-it-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $T | Out-Null

try {
    Write-Host "`n== usage errors (expect exit 2, stderr, empty stdout) =="
    Assert-Cli "unknown flag"         2 "Error: unknown option"             $Bin @('create', '--bogus', "$T/a")
    Assert-Cli "missing path"         2 "Error: a target path is required"  $Bin @('create', '--agent', 'foo')
    Assert-Cli "--agent without name" 2 "Error: --agent requires a name"    $Bin @('create', "$T/b", '--agent')
    Assert-Cli "--workflow no name"   2 "Error: --workflow requires a name" $Bin @('create', "$T/c", '--workflow')
    Assert-Cli "multiple paths"       2 "Error: multiple target paths"      $Bin @('create', "$T/d", "$T/e")
    Assert-Cli "--with-ui workflow"   2 "Error: --with-ui requires at least one agent" $Bin @('create', "$T/f", '--workflow', 'w', '--with-ui')
    Assert-Cli "reserved name"        2 "is reserved"                       $Bin @('create', "$T/g", '--agent', 'ui')
    Assert-Cli "invalid name (slash)" 2 "invalid workforce name"            $Bin @('create', "$T/h", '--agent', 'a/b')
    Assert-Cli "duplicate name"       2 "duplicate workforce member name"   $Bin @('create', "$T/i", '--agent', 'x', '--agent', 'x')
    Assert-Cli "interactive no TTY"   2 "interactive create requires a terminal" $Bin @('create', "$T/j")

    Write-Host "`n== preconditions =="
    New-Item -ItemType Directory -Force -Path "$T/nonempty" | Out-Null
    Set-Content -Path "$T/nonempty/keep" -Value "x"
    Assert-Cli "existing non-empty dir" 2 "already exists and is not empty" $Bin @('create', "$T/nonempty", '--agent', 'foo')

    Write-Host "`n== help (expect exit 0) =="
    Assert-Cli "create -h"            0 "Create a new timbal project"       $Bin @('create', '-h')

    if ($env:TIMBAL_CLI_E2E_NETWORK -eq '1') {
        Write-Host "`n== success (network: fetches blueprints) =="
        Assert-SuccessQuiet $Bin "$T/proj"
        Assert-SuccessWithUi $Bin "$T/projui"
    }
    else {
        Write-Host "`n(skipping success cases; set TIMBAL_CLI_E2E_NETWORK=1 to run them)"
    }
}
finally {
    Remove-Item -Recurse -Force $T -ErrorAction SilentlyContinue
    Remove-Item -Force $EmptyIn -ErrorAction SilentlyContinue
}

Write-Host "`n----------------------------------------"
if ($script:Fail -eq 0) {
    Write-Host "All $($script:Pass) checks passed." -ForegroundColor Green
    exit 0
}
Write-Host "$($script:Fail) failed, $($script:Pass) passed." -ForegroundColor Red
exit 1
