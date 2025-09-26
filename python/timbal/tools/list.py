import os
from pathlib import Path

import structlog

from timbal import Tool

logger = structlog.get_logger("timbal.tools.list_files")


class List(Tool):

    def __init__(self, **kwargs):
        
        async def list_files(
            directory: str,
        ) -> str:
            """
            List files and directories in the specified directory with pattern matching.
            
            Args:
                directory: Directory path to list files from
            
            Returns:
                Formatted string listing files and directories
            """
            try:

                path = Path(os.path.expandvars(os.path.expanduser(directory))).resolve()
                
                if not path.exists():
                    return f"Directory not found: {directory}"
                
                if not path.is_dir():
                    return f"Path is not a directory: {directory}"

                return list(path.iterdir())
                
                
            except Exception as e:
                logger.error("Error listing files", error=str(e), directory=directory)
                return f"Error listing files in {directory}: {str(e)}"
        
        super().__init__(
            name="list_files",
            description="List files and directories",
            handler=list_files,
            **kwargs
        )

