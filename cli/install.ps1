# This script is adapted from the Python uv installer script (uv-installer.ps1)
# found at: https://astral.sh/uv/install.ps1

# The original script is licensed under the MIT License.
# As this script is a derivative work, it is also licensed under the MIT License.


<#
.SYNOPSIS
Installer for the Timbal CLI.

.DESCRIPTION
This script installs the Timbal Command Line Interface (CLI).
It detects the host architecture (Windows x86_64 or aarch64), downloads the appropriate
Timbal executable from the latest GitHub release, and installs it to:
    $env:USERPROFILE\.local\bin\timbal.exe

The script then adds the installation directory ($env:USERPROFILE\.local\bin)
to the user's PATH environment variable, making the 'timbal' command available
in the terminal.

This script requires PowerShell 5 or later and an appropriate Execution Policy
(e.g., RemoteSigned, Bypass).

.PARAMETER Help
Displays this help message.

.NOTES
Source: Adapted from the uv installer script (uv-installer.ps1) found at https://astral.sh/uv/install.ps1
License: MIT License
#>


param (
    [Parameter(HelpMessage = "Print Help")]
    [switch]$Help
)


$InformationPreference = 'Continue'


function Initialize-Environment() {
    If (($PSVersionTable.PSVersion.Major) -lt 5) {
        throw @"
Error: PowerShell 5 or later is required to install timbal.
Upgrade PowerShell:

    https://docs.microsoft.com/en-us/powershell/scripting/setup/installing-windows-powershell

"@
    }

    # show notification to change execution policy:
    $allowedExecutionPolicy = @('Unrestricted', 'RemoteSigned', 'Bypass')
    If ((Get-ExecutionPolicy).ToString() -notin $allowedExecutionPolicy) {
        throw @"
Error: PowerShell requires an execution policy in [$($allowedExecutionPolicy -join ", ")] to run timbal. For example, to set the execution policy to 'RemoteSigned' please run:

    Set-ExecutionPolicy RemoteSigned -scope CurrentUser

"@
    }

    # GitHub requires TLS 1.2
    If ([System.Enum]::GetNames([System.Net.SecurityProtocolType]) -notcontains 'Tls12') {
        throw @"
Error: Installing timbal requires at least .NET Framework 4.5
Please download and install it first:

    https://www.microsoft.com/net/download

"@
    }
}


function Test-UvInstallation() {
    Write-Information "Checking for 'uv' command in PATH..."

    try {
        Get-Command uv -ErrorAction Stop | Out-Null
        Write-Information "'uv' command found successfully."
    } catch [System.Management.Automation.CommandNotFoundException] {
        # Specific catch for command not found
        $errorMessage = @"
'uv' command not found in PATH. Timbal requires 'uv' (from Astral) for Python project management.

Possible Solutions:
1. Install 'uv' using the official installer:
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
2. If 'uv' is already installed, ensure its installation directory
   is correctly added to your User or System PATH environment variable and restart your terminal.
"@
        Write-Error $errorMessage
        Exit 1
    } catch {
        # Catch any other unexpected errors during the check
        Write-Error "An unexpected error occurred while checking for 'uv': $($_.Exception.Message)"
        Exit 1
    }
}


function Test-GitInstallation() {
    Write-Information "Checking for 'git' command in PATH..."

    try {
        Get-Command git -ErrorAction Stop | Out-Null
        Write-Information "'git' command found successfully."
    } catch [System.Management.Automation.CommandNotFoundException] {
        $errorMessage = @"
'git' command not found in PATH. Git is required for version control and credential management with the Timbal Platform.

Possible Solutions:
1. Install git:
   Download from: https://git-scm.com/downloads/win
   Or via winget: winget install Git.Git
2. If git is already installed, ensure its installation directory
   is correctly added to your User or System PATH environment variable and restart your terminal.
"@
        Write-Error $errorMessage
        Exit 1
    } catch {
        Write-Error "An unexpected error occurred while checking for 'git': $($_.Exception.Message)"
        Exit 1
    }
}


function Test-BunInstallation() {
    Write-Information "Checking for 'bun' command in PATH..."

    try {
        Get-Command bun -ErrorAction Stop | Out-Null
        Write-Information "'bun' command found successfully."
    } catch [System.Management.Automation.CommandNotFoundException] {
        $errorMessage = @"
'bun' command not found in PATH. Timbal uses bun to manage packages and run UIs and APIs for your projects.

Possible Solutions:
1. Install 'bun' using the official installer:
   powershell -c "irm bun.sh/install.ps1 | iex"
2. If 'bun' is already installed, ensure its installation directory
   is correctly added to your User or System PATH environment variable and restart your terminal.

For more information, visit: https://bun.sh/docs/installation#windows
"@
        Write-Error $errorMessage
        Exit 1
    } catch {
        Write-Error "An unexpected error occurred while checking for 'bun': $($_.Exception.Message)"
        Exit 1
    }
}


function Get-Arch() {
    try {
        # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
        # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
        # Ideally this would just be
        #   [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
        # but that gets a type from the wrong assembly on Windows PowerShell (i.e. not Core)
        $a = [System.Reflection.Assembly]::LoadWithPartialName("System.Runtime.InteropServices.RuntimeInformation")
        $t = $a.GetType("System.Runtime.InteropServices.RuntimeInformation")
        $p = $t.GetProperty("OSArchitecture")
        # Possible OSArchitecture Values: https://learn.microsoft.com/dotnet/api/system.runtime.interopservices.architecture
        # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
        switch ($p.GetValue($null).ToString())
        {
            "X86" { return "i686-pc-windows" }
            "X64" { return "x86_64-pc-windows" }
            "Arm" { return "thumbv7a-pc-windows" }
            "Arm64" { return "aarch64-pc-windows" }
        }
    } catch {
        # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
        # prior to Windows 10 v1709 may not have this API.
        Write-Verbose "Get-TargetTriple: Exception when trying to determine OS architecture."
        Write-Verbose $_
    }

    # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
    Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
    if ([System.Environment]::Is64BitOperatingSystem) {
        return "x86_64-pc-windows"
    } else {
        return "i686-pc-windows"
    }
}


function Get-Executable($DestinationPath) {
    if ([string]::IsNullOrWhiteSpace($DestinationPath)) {
        throw "Get-Executable requires a non-empty -DestinationPath parameter."
    }

    $arch = Get-Arch
    Write-Information "Detected architecture: $arch"

    if ($arch -ne "x86_64-pc-windows") {
        throw "ERROR: Timbal installation currently only supports the 'x86_64-pc-windows' (amd64) architecture. Detected: $arch"
    }

    $manifestUrl = "https://github.com/timbal-ai/timbal/releases/latest/download/manifest.json"
    $manifestPath = [System.IO.Path]::GetTempFileName()
    try {
        Invoke-WebRequest -Uri $manifestUrl -OutFile $manifestPath -UseBasicParsing -ErrorAction Stop
    } catch {
        throw "Failed to download manifest.json from $manifestUrl. Error $($_.Exception.Message)"
    }

    try {
        $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
        $url = $manifest.binaries.windows.x86_64.url
        if (-not $url) {
            Remove-Item $manifestPath -Force -ErrorAction SilentlyContinue
            throw "No download URL found in manifest for windows and x86_64"
        }
    } catch {
        Remove-Item $manifestPath -Force -ErrorAction SilentlyContinue
        throw "Failed to parse manifest.json. Error $($_.Exception.Message)"
    } finally {
        Remove-Item $manifestPath -Force -ErrorAction SilentlyContinue
    }

    Write-Information "Downloading $url to $DestinationPath"

    try {
        # Ensure TLS 1.2+ is used
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12, [Net.SecurityProtocolType]::Tls13

        # Remove existing destination file if it exists to ensure a clean download
        if (Test-Path -Path $DestinationPath) {
            Write-Warning "Overwriting existing file at '$DestinationPath'."
            Remove-Item -Path $DestinationPath -Force -ErrorAction Stop
        }

        # Download the file directly to the final destination path
        Invoke-WebRequest -Uri $url -OutFile $DestinationPath -UseBasicParsing -ErrorAction Stop

        # Check if the file was actually downloaded and is not empty
        if (-not (Test-Path -Path $DestinationPath) -or (Get-Item $DestinationPath).Length -eq 0) {
             # Clean up potentially corrupt/empty file before throwing
             if (Test-Path -Path $DestinationPath) { Remove-Item -Path $DestinationPath -Force -ErrorAction SilentlyContinue }
             throw "Downloaded file is missing or empty. Check if the release asset exists at $url"
        }

        # No return value needed
    } catch {
        # Attempt cleanup of potentially partial download at the destination
        if (Test-Path -Path $DestinationPath) {
            Remove-Item -Path $DestinationPath -Force -ErrorAction SilentlyContinue
        }
        # Rethrow a more specific error including the intended destination
        throw "FATAL: Failed to download Timbal CLI from '$url' to '$DestinationPath'. Check URL, network, and permissions. Error: $($_.Exception.Message)"
    }
}


function Install-Binary($install_args) {
    if ($Help) {
        Get-Help $PSCommandPath -Detailed
        Exit
    }

    Initialize-Environment

    Test-GitInstallation
    Test-UvInstallation
    Test-BunInstallation

    $InstallDir = Join-Path -Path $env:USERPROFILE -ChildPath ".local\bin"

    if (-not (Test-Path -Path $InstallDir -PathType Container)) {
        Write-Information "Creating installation directory: '$InstallDir'"
        try {
            New-Item -Path $InstallDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
        } catch {
            throw "Failed to create installation directory '$InstallDir'. Error: $($_.Exception.Message)"
        }
    } else {
        Write-Information "Installation directory already exists: '$InstallDir'"
    }

    $CliExecutableName = "timbal.exe"
    $CliExecutablePath = Join-Path -Path $InstallDir -ChildPath $CliExecutableName

    Get-Executable -DestinationPath $CliExecutablePath

    # Add installation directory to the user PATH persistently if not already present
    $CurrentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not ($CurrentUserPath -split ";" | Where-Object { $_ -eq $InstallDir })) {
        $NewUserPath = "$InstallDir;$CurrentUserPath"
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
        Write-Information "Added '$InstallDir' to your user PATH. Restart your terminal for changes to take effect."
    } else {
        Write-Information "'$InstallDir' is already in your user PATH."
    }

    Write-Information "Successfully installed timbal. Run 'timbal configure' to set up your credentials and settings."
}


try {
    Install-Binary "$Args"
} catch {
    Write-Information $_
    exit 1
}
