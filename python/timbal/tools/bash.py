import asyncio
import subprocess
from typing import Any, Literal
from webbrowser import get

import psutil
import structlog
from pydantic import Field

from timbal.state import get_run_context

from ..core.tool import Tool

logger = structlog.get_logger("timbal.core.bash_tool")


class Bash(Tool):
    """A Tool for executing bash commands"""

    def __init__(self, cmd: str, **kwargs):
        async def _execute_cmd(
            args: str = Field(default="", description="Additional arguments to append to the command. Supports shell pipelines (e.g., '| grep .py'). For cmd='*', this becomes the full command to execute."),
            background: bool = Field(default=False, description="Whether to execute in background"),
            capture_output: bool = Field(default=True, description="Whether to capture output"),
            cwd: str | None = Field(default=None, description="Working directory for execution"),
        ) -> dict[str, Any]:
            """Execute the predefined shell command with optional arguments and pipeline support."""
            if self.cmd == "*":
                if not args:
                    return await asyncio.to_thread(lambda: 'Error: When using cmd="*", you must provide the full command in the args parameter')
                full_command = args
            else:
                # Original behavior for specific commands
                additional_parts = args.split() if args else []
                full_command = self.cmd.split() + additional_parts
                            
            return await self._execute_command(full_command, background, capture_output, cwd)
        
        if cmd == "*":
            tool_name = "execute_bash_commands"
            tool_description = "Execute any bash command."
        else:
            tool_name = f"{cmd}_tool"
            tool_description = f"Execute the bash command: {cmd}."
        
        super().__init__(
            name=tool_name,
            description=tool_description,
            handler=_execute_cmd,
            **kwargs
        )
        self.cmd = cmd

    
    async def _manage_background_tasks(self):
        agent = get_run_context().parent_trace().runnable

        if _get_process_output_tool.name not in agent._tools_by_name:
            for tool in [_get_process_output_tool, _terminate_process_tool]:
                # Nest the tools under the agent's path
                tool.nest(agent._path)
                agent.tools.append(tool)
                agent._tools_by_name[tool.name] = tool      


    async def _execute_command(
        self,
        command: list[str] | str = Field(
            description="The command to execute. List of strings (direct execution) or string (shell execution)"
        ),
        background: bool = Field(
            default=False,
            description="Whether to execute the command in the background"
        ),
        capture_output: bool = Field(
            default=True,
            description="Whether to capture stdout and stderr"
        ),
        cwd: str | None = Field(
            default=None,
            description="Working directory for command execution"
        )
    ) -> dict[str, Any]:
        """
        Execute a shell command asynchronously with various execution options.
        
        Args:
            command: Command to execute (string for shell, list for direct execution)
            background: If True, returns immediately after starting the process
            capture_output: If True, captures stdout and stderr
            
        Returns:
            Dictionary containing process information, output, and status
        """
        # Prepare subprocess arguments
        kwargs = {}
        
        if capture_output:
            kwargs['stdout'] = asyncio.subprocess.PIPE
            kwargs['stderr'] = asyncio.subprocess.PIPE

        if cwd:
            kwargs['cwd'] = cwd

        # Use shell=True for commands that need shell expansion or pipelines
        if isinstance(command, list):
            command_str = ' '.join(str(arg) for arg in command)
            command = command_str
        # If command is already a string, use it as-is
         
        try:
            if background:
                # Start process in background
                kwargs['shell'] = True
                process = subprocess.Popen(command, **kwargs)
                get_run_context().parent_trace().runnable._bg_tasks[process.pid] = process
                await self._manage_background_tasks()

                logger.info(
                    "Added background process to _bg_tasks",
                    pid=process.pid,
                    process=command,
                    ALL=get_run_context().parent_trace().runnable._bg_tasks
                )    
                return {
                    'pid': process.pid,
                    'background': True,
                    # 'process': process,
                    'message': f'Process started with PID {process.pid}'
                }

            # Not background execution
            process = await asyncio.create_subprocess_shell(
                command,
                **kwargs
            )

            # Wait for completion and capture output
            stdout, stderr = await process.communicate()
            
            # Decode bytes to strings if output was captured
            if capture_output:
                stdout = stdout.decode('utf-8') if stdout else ''
                stderr = stderr.decode('utf-8') if stderr else ''
            
            return {
                'background': False,
                'output': stdout,
                'stderr': stderr,
                'returncode': process.returncode
            }

        except Exception as e:
            return {
                'command': command,
                'message': f'Failed to execute command: {str(e)}',
            }


async def _get_process_output(process_pid: int = Field(description="Process ID to get output from")) -> dict[str, Any]:
    """
    Get the output of a running process.
    
    Args:
        process_pid: Process ID to get output from
    """
    logger.info(
        "INFO",
        ALL=get_run_context().parent_trace().runnable._bg_tasks
    )  
    try:
        process = get_run_context().parent_trace().runnable._bg_tasks[process_pid]
        status = process.poll()
        # Not finished yet
        if status is None:
            return {
                'pid': process_pid,
                'output': "Process is still running"
            }
        stdout, stderr = process.communicate()
        # Terminate process (avoid zombie)
        del get_run_context().parent_trace().runnable._bg_tasks[process_pid]
        # Remove process
        return {
            'pid': process_pid,
            'output': stdout,
            'error': stderr
        }
    except Exception as e:  
        if get_run_context().parent_trace().runnable._bg_tasks[process_pid]:
            del get_run_context().parent_trace().runnable._bg_tasks[process_pid]
        return {
            'pid': process_pid,
            'message': f'Failed to get process output: {str(e)}'
        }

async def _check_background_tasks():
    agent = get_run_context().parent_trace().runnable
    if not agent._bg_tasks and _get_process_output_tool.name in agent._tools_by_name:
        for tool in [_get_process_output_tool, _terminate_process_tool]:
            agent.tools.remove(tool)
            del agent._tools_by_name[tool.name]

async def _terminate_process(
    pid: int = Field(description="Process ID to terminate"),
    method: Literal["kill", "terminate"] = Field(
        default="terminate",
        description="Method to terminate the process"
    ),
) -> dict[str, Any]:
    """
    Terminate a running process.
    
    Args:
        pid: Process ID to terminate
        method: Method to terminate the process
        
    Returns:
        Termination result
    """
    try:
        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            if method == "kill":
                process.kill()
                method = 'killed'
            else:
                process.terminate()
                method = 'terminated'
            # Remove process from _bg_tasks
            del get_run_context().parent_trace().runnable._bg_tasks[pid]
            # Check if there are background tasks
            await _check_background_tasks()
            
            return f'Process {pid} {method} successfully'
        else:
            return f'Process {pid} not found'
    except Exception as e:
        return f'Error terminating process {pid}: {str(e)}'


# Private tools for managing background tasks
_terminate_process_tool = Tool(
    name="terminate_process",
    description="Use it always to terminate or kill a process.",
    handler=_terminate_process
)

_get_process_output_tool = Tool(
    name="get_process_output", 
    description="Get the output of a running process.",
    handler=_get_process_output
)
