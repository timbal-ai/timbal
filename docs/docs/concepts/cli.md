---
title: 'CLI'
sidebar: 'docsSidebar'
---

# Timbal CLI Documentation

The Timbal CLI provides a set of commands to interact with Timbal, allowing you to create, run and build flows.

## Installation

To use the Timbal CLI, make sure you have Timbal installed:

```shell
pip install timbal
```

## Usage

The basic structure of a Timbal CLI command is:

```shell
timbal [COMMAND] [OPTIONS] [ARGUMENTS]
```

# Available Commands

### `init`

Initialize a new Timbal project in the specified directory.

```shell
timbal init [PATH]
```


**Options:**
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information
- `-V, --version` - Display version information

Creates a new project with the following structure:

```
my-project/
├── flow.py
├── .dockerignore
├── pyproject.toml
└── timbal.yaml
```


### `build`

Build a container for the application ready for deployment.

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

## `run`

Run a command inside the built container.

```shell
timbal run [OPTIONS] IMAGE [COMMAND]
```


**Options:**
- `-d, --detach` - Run container in the background and print container ID
- `-p, --publish` - Publish a container's port to the host (e.g., -p 8000)
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information

### `push`

Push an application to the Timbal Platform.

```shell
timbal push [OPTIONS] IMAGE
```


**Options:**
- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help information

**Environment Variables:**
- `TIMBAL_API_TOKEN` - Required for authentication with the Timbal Platform

### Global Options

These options are available for all commands:

- `-q, --quiet` - Do not print any output
- `-v, --verbose` - Use verbose output
- `-h, --help` - Display help for the command
- `-V, --version` - Display the Timbal version

## Configuration

Timbal uses a `timbal.yaml` file for project configuration. Here's an example:

```yaml
build:

  # A list of ubuntu apt packages to install.
  # system_packages:
  #   - "libgl1-mesa-glx"
  #   - "libglib2.0-0"

  # Commands run after the environment is setup.
  # run:
  #   - "echo env is ready!"
  #   - "echo another command if needed"


# Path to the flow to be run in the application.
flow: "flow.py:flow"

```


The configuration file allows you to:
- Specify system packages to be installed in the container
- Define the flow file and entry point
- Configure build-time settings

For more detailed information about each command, use the `--help` flag:

```shell
timbal [COMMAND] --help
```
