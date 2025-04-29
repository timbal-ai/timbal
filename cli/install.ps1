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


function Test-DockerInstallation() {
    Write-Information "Checking Docker installation..."

    try {
        Get-Command docker -ErrorAction Stop | Out-Null
        Write-Information "Docker command found successfully."
    } catch [System.Management.Automation.CommandNotFoundException] {
        # Specific catch for command not found
        $errorMessage = @"
Docker command not found in PATH. Timbal requires Docker for container management.

Please follow the instructions in: https://docs.docker.com/desktop/setup/install/windows-install/
"@
        Write-Error $errorMessage
        Exit 1
    } catch {
        # Catch any other unexpected errors during the check
        Write-Error "An unexpected error occurred while checking for 'docker': $($_.Exception.Message)"
        Exit 1
    }

    # Check if docker command is usable by running hello-world
    Write-Information "Checking Docker connectivity by running 'hello-world' container..."
    # Attempt to run hello-world, suppress output, and check exit status
    # *> $null redirects both stdout (1) and stderr (2) streams to null
    docker run --rm hello-world *> $null
    if ($LASTEXITCODE -ne 0) {
        # Use Write-Warning for non-critical issues
        $errorMessage = @"
Docker engine appears to be not running, or the current user lacks permissions.
Failed to run the 'hello-world' container (Exit Code: $LASTEXITCODE).
You might need to start Docker Desktop or configure user permissions.
For Windows, ensure Docker Desktop is running. For Linux/WSL, you might need to add your user to the 'docker' group.
See relevant documentation: https://docs.docker.com/desktop/setup/install/windows-install/
"@
        Write-Error $errorMessage
        Exit 1
    } else {
        Write-Information "Docker connection successful ('hello-world' container ran successfully)."
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
        Write-ErrorAndExit "Get-Executable requires a non-empty -DestinationPath parameter."
    }

    $arch = Get-Arch
    Write-Information "Detected architecture: $arch"

    if ($arch -ne "x86_64-pc-windows") {
        Write-ErrorAndExit "ERROR: Timbal installation currently only supports the 'x86_64-pc-windows' (amd64) architecture. Detected: $arch"
    }

    $url = "https://github.com/timbal-ai/timbal/releases/latest/download/timbal-Windows-x86_64.exe"
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
             Write-ErrorAndExit "Downloaded file is missing or empty. Check if the release asset exists at $url"
        }

        Write-Information "Download and save successful to '$DestinationPath'."
        # No return value needed
    } catch {
        # Attempt cleanup of potentially partial download at the destination
        if (Test-Path -Path $DestinationPath) {
            Remove-Item -Path $DestinationPath -Force -ErrorAction SilentlyContinue
        }
        # Rethrow a more specific error including the intended destination
        Write-ErrorAndExit "FATAL: Failed to download Timbal CLI from '$url' to '$DestinationPath'. Check URL, network, and permissions. Error: $($_.Exception.Message)"
    }
}


function Install-Binary($install_args) {
    if ($Help) {
        Get-Help $PSCommandPath -Detailed
        Exit
    }

    Initialize-Environment

    Test-UvInstallation
    # TODO Make this step optional. This will only be required if you want to use timbal build or timbal push.
    # TODO Also issue a warning that for authenticating to the registry, one might check the .docker/config.json file and update key "credsStore": "".
    Test-DockerInstallation

    $InstallDir = Join-Path -Path $env:USERPROFILE -ChildPath ".local\bin"

    if (-not (Test-Path -Path $InstallDir -PathType Container)) {
        Write-Information "Creating installation directory: '$InstallDir'"
        try {
            New-Item -Path $InstallDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
        } catch {
            Write-ErrorAndExit "Failed to create installation directory '$InstallDir'. Error: $($_.Exception.Message)"
        }
    } else {
        Write-Information "Installation directory already exists: '$InstallDir'"
    }

    $CliExecutableName = "timbal.exe"
    $CliExecutablePath = Join-Path -Path $InstallDir -ChildPath $CliExecutableName

    Get-Executable -DestinationPath $CliExecutablePath

    # TODO Add timbal to the path
    # To add C:\Users\Casa\.local\bin to your PATH, either restart your shell or run:
    # set Path=C:\Users\Casa\.local\bin;%Path%   (cmd)
    # $env:Path = "C:\Users\Casa\.local\bin;$env:Path"   (powershell)
}


try {
    Install-Binary "$Args"
} catch {
    Write-Information $_
    exit 1
}
