#!/bin/sh

# This script should be run via curl:
#   sh -c "$(curl -fsSL https://raw.githubusercontent.com/timbal-ai/timbal/main/cli/install.sh)"
# or via wget:
#   sh -c "$(wget -qO- https://raw.githubusercontent.com/timbal-ai/timbal/main/cli/install.sh)"
# or via fetch:
#   sh -c "$(fetch -o - https://raw.githubusercontent.com/timbal-ai/timbal/main/cli/install.sh)"
#
# As an alternative, you can first download the install script and run it afterwards:
#   wget https://raw.githubusercontent.com/timbal-ai/timbal/main/cli/install.sh
#   sh install.sh

# By default, timbal will be installed at ~/.local/bin/timbal


# This install script is based on that of ohmyzsh[1], which is licensed under the MIT License
# [1] https://github.com/ohmyzsh/ohmyzsh/blob/master/tools/install.sh
# MIT License

# Copyright (c) 2009-2022 Robby Russell and contributors (https://github.com/ohmyzsh/ohmyzsh/contributors)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Exit immediately if a command exits with a non-zero status
set -e


# Parse command line arguments
AUTO_YES=false
for arg in "$@"; do
    case "$arg" in
        -y|--yes)
            AUTO_YES=true
            ;;
    esac
done


# Function to report errors and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}


set_install_dir() {
    DEFAULT_INSTALL_DIR="$HOME/.local/bin"

    if [ -z "$HOME" ]; then
        error_exit "\$HOME env variable is not set. Cannot determine default install location."
    fi

    # If INSTALL_DIR is not set, use the default
    if [ -z "${INSTALL_DIR}" ]; then
        INSTALL_DIR="$DEFAULT_INSTALL_DIR"
        echo "Installing timbal to default location: $INSTALL_DIR"
    else
        echo "Installing timbal to user-specified location: $INSTALL_DIR"
    fi

    # Expand INSTALL_DIR to an absolute path
    # Use a subshell for 'cd' to avoid changing the script's working directory
    INSTALL_DIR_EXPANDED=$(cd "$(dirname "$INSTALL_DIR")" && pwd)/$(basename "$INSTALL_DIR") || error_exit "Failed to resolve path: $INSTALL_DIR"
    INSTALL_DIR="$INSTALL_DIR_EXPANDED"

    # Create the directory if it doesn't exist
    # Use mkdir -p to create parent directories as needed and avoid errors if it already exists
    mkdir -p "$INSTALL_DIR" || error_exit "Failed to create installation directory: $INSTALL_DIR. Check permissions."

    # Check if the directory is writable
    if [ ! -w "$INSTALL_DIR" ]; then
        error_exit "Installation directory $INSTALL_DIR is not writable by the current user."
    fi

    # Expand INSTALL_DIR again *after* creation, handling potential symlinks etc.
    # This ensures the path used later (e.g., for PATH modification) is canonical.
    INSTALL_DIR=$(cd "$INSTALL_DIR"; pwd) || error_exit "Failed to get absolute path for: $INSTALL_DIR"

    # Define the final location of the binary
    CLI_EXECUTABLE_PATH="${INSTALL_DIR}/timbal"
}


command_exists() {
    command -v "$@" >/dev/null 2>&1
}


check_uv() {
    if ! command_exists uv; then
        echo "uv is not installed. Timbal requires uv for Python project management."
        if [ "$AUTO_YES" = true ]; then
            echo "Auto-installing uv (--yes flag provided)..."
            install_uv
        else
            read -p "Install uv now? (Y/n): " choice
            case "$choice" in
                n|N ) echo "Warning: uv is required for Timbal to work. You can install it later from https://docs.astral.sh/uv/";;
                * ) install_uv;;
            esac
        fi
    fi
}


install_uv() {
    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || error_exit "Failed to install uv."
    elif command_exists wget; then
        wget -qO- https://astral.sh/uv/install.sh | sh || error_exit "Failed to install uv."
    else
        error_exit "Cannot install uv: curl or wget is required. Please install uv manually from https://docs.astral.sh/uv/"
    fi
    # Re-source the shell environment to pick up uv in PATH
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) export PATH="$HOME/.local/bin:$PATH" ;;
    esac
    if ! command_exists uv; then
        error_exit "uv was installed but could not be found in PATH. Please restart your terminal and re-run this script."
    fi
    echo "uv installed successfully."
}


check_git() {
    if ! command_exists git; then
        error_exit "git is not installed or not in PATH.
Git is required for version control and credential management with the Timbal Platform.

Install git:
    macOS:  brew install git
    Ubuntu/Debian: sudo apt install git
    Fedora: sudo dnf install git
    Other:  https://git-scm.com/downloads"
    fi
}


check_bun() {
    if ! command_exists bun; then
        echo "bun is not installed. Timbal uses bun to manage packages and run UIs and APIs for your projects."
        if [ "$AUTO_YES" = true ]; then
            echo "Auto-installing bun (--yes flag provided)..."
            install_bun
        else
            read -p "Install bun now? (Y/n): " choice
            case "$choice" in
                n|N ) echo "Warning: bun is required to run UIs and APIs with Timbal. You can install it later from https://bun.sh";;
                * ) install_bun;;
            esac
        fi
    fi
}


install_bun() {
    if command_exists curl; then
        curl -fsSL https://bun.sh/install | bash || error_exit "Failed to install bun."
    elif command_exists wget; then
        wget -qO- https://bun.sh/install | bash || error_exit "Failed to install bun."
    else
        error_exit "Cannot install bun: curl or wget is required. Please install bun manually from https://bun.sh"
    fi
    # Re-source the shell environment to pick up bun in PATH
    bun_bin="${BUN_INSTALL:-$HOME/.bun}/bin"
    case ":$PATH:" in
        *":${bun_bin}:"*) ;;
        *) export PATH="${bun_bin}:$PATH" ;;
    esac
    if ! command_exists bun; then
        error_exit "bun was installed but could not be found in PATH. Please restart your terminal and re-run this script."
    fi
    echo "bun installed successfully."
}


download_file() {
    URL="$1"
    DESTINATION="$2"

    echo "Downloading $URL to $DESTINATION..."

    if command_exists curl; then
        curl --fail --location --output "$DESTINATION" "$URL" || error_exit "curl download failed"
    elif command_exists wget; then
        wget --quiet --output-document="$DESTINATION" "$URL" || error_exit "wget download failed"
    elif command_exists fetch; then
        fetch --quiet --output="$DESTINATION" "$URL" || error_exit "fetch download failed"
    else
        error_exit "Cannot download: curl, wget, or fetch is required."
    fi

    # Basic check if download seems successful (e.g., didn't download a "Not Found" HTML page)
    # A more robust check would involve checking file size or type if possible
    if grep -q "Not Found" "$DESTINATION" && [ "$(wc -c < "$DESTINATION")" -lt 1024 ]; then
        error_exit "File not found or download failed at $URL. Check if url $URL is correct."
        # Clean up the invalid file
        rm -f "$DESTINATION"
        # The error_exit above already stops the script, but keep rm for clarity
    fi
}


setup_timbal() {
    # Real paths are:
    # <name>-<version>-<os>-<arch>(-<abi/toolchain>...).<ext>
    # We will fetch the manifest to get the exact release url for our os and arch.
    MANIFEST_URI="https://github.com/timbal-ai/timbal/releases/latest/download/manifest.json"
    MANIFEST_PATH=$(mktemp)
    trap 'rm -f "$MANIFEST_PATH"' EXIT TERM INT

    download_file "$MANIFEST_URI" "$MANIFEST_PATH"

    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')

    # Parse the URL from the manifest without jq.
    # The manifest is pretty-printed JSON, so we find the OS section,
    # then within it find the arch, then grab the url on the next line.
    DOWNLOAD_URL=$(sed -n "/\"${OS}\"/,/^        }/p" "$MANIFEST_PATH" \
        | sed -n "/\"${ARCH}\"/,/}/p" \
        | grep '"url"' \
        | head -1 \
        | sed 's/.*"url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

    if [ -z "$DOWNLOAD_URL" ] || [ "$DOWNLOAD_URL" = "null" ]; then
        error_exit "No download URL found for OS: $OS, ARCH: $ARCH"
    fi

    echo "Binary download URL: $DOWNLOAD_URL"
    echo "Target executable path: $CLI_EXECUTABLE_PATH"

    # Check if the file already exists
    if [ -e "$CLI_EXECUTABLE_PATH" ]; then
        echo "A file already exists at $CLI_EXECUTABLE_PATH"
        if [ "$AUTO_YES" = true ]; then
            echo "Auto-accepting overwrite (--yes flag provided)"
            rm -f "$CLI_EXECUTABLE_PATH" || error_exit "Failed to remove existing file at $CLI_EXECUTABLE_PATH"
        else
            # Ask if user wants to overwrite. Default to No.
            read -p "Overwrite? (y/N): " choice
            case "$choice" in
                y|Y )
                    echo "Overwriting existing file..."
                    rm -f "$CLI_EXECUTABLE_PATH" || error_exit "Failed to remove existing file at $CLI_EXECUTABLE_PATH"
                    ;;
                * )
                    echo "Skipping installation because file exists."
                    return
                    ;;
            esac
        fi
    fi

    download_file "$DOWNLOAD_URL" "$CLI_EXECUTABLE_PATH"

    chmod +x "$CLI_EXECUTABLE_PATH" || error_exit "Failed to set execute permission on $CLI_EXECUTABLE_PATH"

    # macOS Gatekeeper Fix (if needed)
    if [ "$OS" = "Darwin" ]; then
        if command_exists xattr; then
            xattr -d com.apple.quarantine "$CLI_EXECUTABLE_PATH" 2>/dev/null || echo "Info: Failed to remove quarantine attribute (this is common if it wasn't set). Gatekeeper might still prompt on first run."
        fi
    fi

    # Add install directory to PATH if it's not already there
    setup_path
}


setup_path() {
    SHELL_NAME=$(basename "$SHELL")
    PROFILE_FILE=""

    # Determine the likely profile file
    if [ "$SHELL_NAME" = "bash" ]; then
        # Check common bash profile files
        if [ -f "$HOME/.bashrc" ]; then
            PROFILE_FILE="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            PROFILE_FILE="$HOME/.bash_profile"
        elif [ -f "$HOME/.profile" ]; then # Fallback for login shells
             PROFILE_FILE="$HOME/.profile"
        fi
    elif [ "$SHELL_NAME" = "zsh" ]; then
        # Zsh typically uses .zshrc
        if [ -f "$HOME/.zshrc" ]; then
            PROFILE_FILE="$HOME/.zshrc"
        elif [ -f "$HOME/.zprofile" ]; then # Fallback for login shells
            PROFILE_FILE="$HOME/.zprofile"
        fi
    elif [ -f "$HOME/.profile" ]; then # Generic fallback for other sh-like shells
        PROFILE_FILE="$HOME/.profile"
    fi

    # Check if INSTALL_DIR is already in PATH
    # Use grep -q for quiet check; use escaped ':' for delimiters
    if echo ":$PATH:" | grep -q ":${INSTALL_DIR}:"; then
        echo "$INSTALL_DIR is already in your PATH."
    elif [ -z "$PROFILE_FILE" ]; then
        echo "Warning: Could not automatically find shell profile file (e.g., .bashrc, .zshrc, .profile)."
        echo "Please add the following line to your shell profile manually:"
        echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    else
        echo "Adding $INSTALL_DIR to PATH in $PROFILE_FILE..."
        # Check if the specific export line or a comment marker already exists
        if grep -q "# Added by $CLI_NAME install script" "$PROFILE_FILE" || grep -q "export PATH=.*$INSTALL_DIR" "$PROFILE_FILE"; then
             echo "$INSTALL_DIR seems to be already configured in $PROFILE_FILE."
        else
            # Add the export line, preceded by a comment
            echo "" >> "$PROFILE_FILE" # Add a newline for separation
            echo "# Added by $CLI_NAME install script on $(date)" >> "$PROFILE_FILE"
            echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$PROFILE_FILE" || error_exit "Failed to write to $PROFILE_FILE"
            echo "Successfully added path to $PROFILE_FILE."
            echo "Please restart your terminal or run 'source \"$PROFILE_FILE\"' to update your PATH."
        fi
    fi
}


main() {
    set_install_dir

    if command_exists timbal; then
        echo "A timbal command already exists on your system at the following location: $(which timbal)"
        if [ "$AUTO_YES" = true ]; then
            echo "Overwriting..."
        else
            echo "The installations may interfere with one another."
            echo "Do you want to continue with this installation anyway?"
            read -p "Continue? (y/N): " choice
            case "$choice" in
                y|Y ) echo "Continuing with installation...";;
                * ) echo "Exiting installation."; exit 1;;
            esac
        fi
    fi

    check_git
    check_uv
    check_bun

    setup_timbal

    if command_exists timbal; then
        echo "Successfully installed timbal. Run 'timbal configure' to set up your credentials and settings."
    else
        echo 'Error: timbal not installed.'
        exit 1
    fi
}

main "$@"
