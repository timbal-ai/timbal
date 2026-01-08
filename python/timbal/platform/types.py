"""Internal platform API types.

This module contains Pydantic models for internal communication with the Timbal platform API.
These types are not part of the public API and should not be used directly by end users.
"""

from datetime import datetime

from pydantic import BaseModel


class UploadFileResponse(BaseModel):
    """Response model for file upload endpoints.

    This model represents the response returned by the platform API when uploading
    a file. It contains metadata about the uploaded file including its unique ID,
    storage URL, and content information.

    Note: This is an internal model used for platform API communication.
    End users should interact with files through the File class in timbal.types.file.

    Attributes:
        content_length: Size of the uploaded file in bytes
        content_type: MIME type of the file (e.g., "application/pdf", "image/png")
        created_at: Timestamp when the file was uploaded to the platform
        expires_at: Optional expiration timestamp for temporary files (None for permanent files)
        name: Unique filename assigned by the platform (may differ from original filename)
        id: Unique identifier for the file in the platform's storage system
        url: Full URL where the file can be accessed (typically a CloudFront or S3 URL)
    """

    content_length: int
    content_type: str
    created_at: datetime
    expires_at: datetime | None
    name: str
    id: str | int
    url: str
