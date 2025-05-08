---
title: 'CLI'
sidebar: 'docsSidebar'
---

# Timbal CLI Guide

Welcome to the Timbal Command Line Interface! This is your gateway to building, running, and deploying AI applications with ease!

## Installation

Getting started is super simple! Just install Timbal using pip:

```shell
pip install timbal
```

## Basic Usage

The Timbal CLI follows this simple pattern:

```shell
timbal [COMMAND] [OPTIONS] [ARGUMENTS]
```

## Available Commands

### `init` - Start a New Project

Create a fresh Timbal project in your chosen directory:

```shell
timbal init [PATH]
```

**Options:**
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information
- `-V, --version` - Display version information

This creates a new project with this structure:

```
my-project/
├── flow.py        # Your AI flow code
├── .dockerignore  # Docker ignore rules
├── pyproject.toml # Python project config
└── timbal.yaml    # Timbal configuration
```

### `build` - Package Your App

Build a container for deployment:

```shell
timbal build [OPTIONS] [PATH]
```

**Options:**
- `-t, --tag` - The tag to use for the container
- `--progress` - Set type of progress output ("auto", "quiet", "plain", "tty", "rawjson")
- `--no-cache` - Do not use cache when building the image
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information

### `run` - Execute Your App

Run your application in a container:

```shell
timbal run [OPTIONS] IMAGE [COMMAND]
```

**Options:**
- `-d, --detach` - Run container in the background and print container ID
- `-p, --publish` - Publish a container's port to the host (e.g., -p 8000)
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information

### `push` - Deploy to Platform

Push your app to the Timbal Platform:

```shell
timbal push [OPTIONS] IMAGE
```

**Options:**
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information

**Required Environment:**
- `TIMBAL_API_TOKEN` - Your platform authentication token

## Global Options

These options work with all commands:

- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help for the command
- `-V, --version` - Display the Timbal version

## Configuration

Customize your project using `timbal.yaml`:

```yaml
build:
  # Install system packages
  # system_packages:
  #   - "libgl1-mesa-glx"
  #   - "libglib2.0-0"

  # Run setup commands
  # run:
  #   - "echo env is ready!"
  #   - "echo another command if needed"

# Your flow entry point
flow: "flow.py:flow"
```

### Configuration Features:

1. **System Setup**
   - Install required packages
   - Run setup commands
   - Configure environment

2. **Flow Configuration**
   - Specify flow file
   - Set entry point
   - Configure runtime

3. **Build Settings**
   - Customize container
   - Set build options
   - Configure deployment

## Pro Tips

1. **Get Help Anywhere**
   ```shell
   timbal [COMMAND] --help
   ```

2. **Debug Like a Pro**
   ```shell
   timbal -v [COMMAND]  # Verbose mode
   ```

3. **Silent Mode**
   ```shell
   timbal -q [COMMAND]  # Quiet mode
   ```

## Want to Learn More?

Check out these related concepts:
- Flows: Learn how to create AI workflows
- Servers: Deploy your applications
- Enterprise: Advanced deployment features
